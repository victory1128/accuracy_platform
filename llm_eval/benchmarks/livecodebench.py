"""LiveCodeBench: 竞赛代码生成评测 (LeetCode 风格)

442 道题, 每题含题面 + 函数名 (entry_point) + 函数式测试用例。
test_code 是 JSON 串: [{"input": "<每行一个JSON参数, \\n分隔>", "output": "<期望返回JSON>",
                        "testtype": "functional"}]

评分: 解析测试用例, 把每个 input 的若干行各解析为 JSON 参数, 调用模型补全的函数,
比对返回值与 output (JSON 解析后相等即通过)。与 HumanEval 的 assert 式不同, 需自定义 eval。
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from ..scoring import extract_code
from ..scoring.code_exec import run_code_tests
from .base import Benchmark
from .registry import register
from ._data import data_path

LCB_HEADER = (
    "Solve the programming problem below. Write a complete, self-contained Python function "
    "with the exact signature given in the problem. Output only the code in a ```python``` block, "
    "including all needed imports. The function will be called with the test inputs and its return "
    "value compared to the expected output."
)


@register
class LiveCodeBench(Benchmark):
    META = BenchmarkMeta(
        name="livecodebench",
        display_name="LiveCodeBench",
        stage=Stage.PRETRAIN,
        task_type=TaskType.CODE,
        description="竞赛代码生成 (LeetCode风格), 442题, 函数式测试 pass@1",
        tags=["modern", "code", "reasoning", "hard"],
        num_fewshot=0,
        source="Jain et al. 2024",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("livecodebench.jsonl")
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            sid = r.get("sample_id") or ""
            samples.append(
                Sample(
                    sample_id=sid,
                    prompt="",
                    question=r.get("prompt", ""),
                    gold="",
                    test_code=r.get("test_code", ""),
                    meta={"entry_point": r.get("entry_point", "")},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        # 明确告知函数名: LiveCodeBench 的 entry_point 多为驼峰 (findPeaks/sumOfSquares),
        # 模型易自行改成蛇形 (find_peaks), 而 harness 按 entry_point 调用 → NameError 误判。
        # 在题面后强制要求用 entry_point 命名, 治本 (执行层 _alias_entry_point 再兜底)。
        ep = sample.meta.get("entry_point", "")
        name_hint = f"\n\nThe function must be named exactly `{ep}`. Define it as `def {ep}(...):`." if ep else ""
        return f"{LCB_HEADER}\n\n{sample.question}{name_hint}\n\n```python\n"

    def parse_params(self) -> dict:
        # 竞赛题思维链模型会先长时间思考, 需足够预算让思考+代码都输出, 否则截断。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        completion = extract_code(response)
        entry_point = sample.meta.get("entry_point", "")
        test_cases = _parse_lcb_tests(sample.test_code or "")

        passed = False
        err = ""
        if completion and entry_point and test_cases:
            harness = _build_lcb_harness(entry_point, test_cases)
            passed, err = run_code_tests(completion, harness, entry_point=entry_point)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=completion[:200] if completion else "",
            correct=passed,
            score=1.0 if passed else 0.0,
            error=err or None,
        )


def _parse_lcb_tests(test_code: str) -> List[dict]:
    """把 test_code (JSON 串) 解析为 [{"args":[...], "expected": ...}, ...]。"""
    if not test_code or not test_code.strip():
        return []
    try:
        raw = json.loads(test_code)
    except (json.JSONDecodeError, TypeError):
        return []
    cases = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        inp = tc.get("input", "")
        # input 每行一个 JSON 参数
        args = []
        for line in str(inp).split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                args.append(json.loads(line))
            except json.JSONDecodeError:
                args.append(line)
        try:
            expected = json.loads(tc.get("output", ""))
        except (json.JSONDecodeError, TypeError):
            expected = tc.get("output", "")
        cases.append({"args": args, "expected": expected})
    return cases


def _build_lcb_harness(entry_point: str, cases: List[dict]) -> str:
    """生成测试 harness: 调用 entry_point(*args), 断言返回值 == expected。

    harness 作为 test_code 与模型补全拼接执行 (run_code_tests 会拼 completion + harness)。
    用 json 序列化比较, 兼容 list/dict 等返回类型。
    """
    lines = ["import json as _json", ""]
    for i, c in enumerate(cases):
        args_repr = _json_safe_repr(c["args"])
        exp_repr = _json_safe_repr(c["expected"])
        lines.append(f"_args{i} = {args_repr}")
        lines.append(f"_exp{i} = {exp_repr}")
        lines.append(f"_got{i} = {entry_point}(*_args{i})")
        lines.append(
            "assert _json.dumps(_got%d, sort_keys=True) == _json.dumps(_exp%d, sort_keys=True), "
            "'case %d failed: got ' + repr(_got%d) + ' want ' + repr(_exp%d)" % (i, i, i, i, i)
        )
    return "\n".join(lines)


def _json_safe_repr(obj) -> str:
    """把 Python 对象转成可 eval 的字面量 (用 json.dumps, 保证类型无损)。"""
    return json.dumps(obj, ensure_ascii=False)
