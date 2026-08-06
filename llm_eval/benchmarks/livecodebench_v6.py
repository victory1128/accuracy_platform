"""LiveCodeBench-v6: 竞赛代码生成评测 (release_v6 全量累积, 官方 1055 题中可评测 1054 题)

数据源 livecodebench/code_generation_lite 的 release_v6 全量累积 (test+test2..test6 共6文件,
官方累积 1055 题; 实测含 1 条既非 functional 也非 stdin 的 other 题跳过, 可评测 1054 题)。
该全集是混合模式, 按 testtype 分类全保留 (exec_mode 字段标注):
  - 函数式 444 题 (LeetCode 风格 "class Solution: def method(self,...)"): completion+harness
    拼一坨跑 1 次, 比对返回值 (run_code_tests)
  - stdin/stdout 式 610 题 (AtCoder/Codeforces 风格): 模型输出完整程序, 对每 case 独立跑、
    喂 stdin、规范化比对 stdout (run_code_stdin)
与官方 release_v6 口径一致 (非仅 v6 增量)。

与原版 LiveCodeBench (test_generation, 纯 "def fn(...)" 函数式) 的关键差异:
  - 无 function_name 字段 → entry_point 从 starter_code 正则提取方法名
  - starter_code 是 class Solution 方法 (带 self) → harness 用 Solution().method(*args),
    而非父类的 fn(*args)。故 evaluate 不能直接继承, 需自定义 harness builder。
  - test 在 public_test_cases 字段 (非 test 字段), 但 JSON 结构相同, 复用 _parse_lcb_tests。

评分 (函数式): 解析 test 用例, 把每个 input 的若干行各解析为 JSON 参数, 实例化 Solution() 调用
方法, 比对返回值与 output (JSON 解析后相等即通过)。
评分 (stdin): 对每个 case 跑被测程序、喂 input、规范化 (去末尾换行/行尾空白) 比对 stdout。
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from ..scoring import extract_code
from ..scoring.code_exec import CODE_BLOCK_RE, run_code_stdin, run_code_tests
from .base import Benchmark
from .registry import register
from ._data import data_path
# 复用 LiveCodeBench 的测试用例解析器 (test_code JSON 串 → [{args, expected}])
from .livecodebench import _parse_lcb_tests, LCB_HEADER


# stdin 式题的 prompt 头: 要求模型写完整可执行程序 (读 stdin 写 stdout), 不要定义函数。
STDIN_HEADER = (
    "Solve the programming problem below. Write a complete, self-contained Python program that "
    "reads from standard input (stdin) and writes to standard output (stdout). Output only the "
    "code in a ```python``` block, including all needed imports. Do not define a function; write "
    "top-level code that runs when executed."
)


@register
class LiveCodeBenchV6(Benchmark):
    META = BenchmarkMeta(
        name="livecodebench_v6",
        display_name="LiveCodeBench-v6",
        stage=Stage.PRETRAIN,
        task_type=TaskType.CODE,
        description="竞赛代码生成 release_v6 全量 (1054题: 函数式444 + stdin610) pass@1",
        tags=["modern", "code", "reasoning", "hard", "contamination_free"],
        num_fewshot=0,
        source="Jain et al. 2024 (release_v6 累积)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("livecodebench_v6.jsonl")
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
                    meta={
                        "entry_point": r.get("entry_point", ""),
                        "starter_code": r.get("starter_code", ""),
                        "exec_mode": r.get("exec_mode", "functional"),
                    },
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        if sample.meta.get("exec_mode") == "stdin":
            # stdin 式题: 要求完整程序读 stdin 写 stdout, 不给函数签名。
            return f"{STDIN_HEADER}\n\n{sample.question}\n\n```python\n"
        # 函数式题 (默认): 明确告知函数名, 防模型把驼峰方法名改成蛇形致 NameError。
        # v6 题目是 class Solution 方法, 额外给出 starter_code 签名让模型按签名补全。
        ep = sample.meta.get("entry_point", "")
        sc = sample.meta.get("starter_code", "")
        name_hint = f"\n\nThe function must be named exactly `{ep}`." if ep else ""
        starter = f"\n\nUse this signature:\n```python\n{sc}\n```" if sc else ""
        return f"{LCB_HEADER}\n\n{sample.question}{name_hint}{starter}\n\n```python\n"

    def parse_params(self) -> dict:
        # 竞赛代码题: 思维链模型 (如 glm-5.2) 会先长时间推理再输出代码, 思维链动辄上万
        # token。8192 会被思维链耗尽导致代码未输出即截断 (finish_reason=length, 无代码块),
        # 全判 fail。用 131072 (与 AIME/HMMT/IMO 等竞赛题一致), 给推理+代码足够预算。
        return {"temperature": 0.0, "max_tokens": 131072, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        completion = extract_code(response)
        # 思维链泄漏检测: prompt 明确要求模型用 ```python 块输出代码。若 response 非空却
        # 无任何代码块, 说明模型把推理写进了 content (思维链泄漏) 且未产出代码 —— 常见于
        # glm-5.2 对部分竞赛题推理发散, 在 max_tokens (131072) 耗尽前没开始写代码
        # (finish_reason=length, response 是数十万字符的自然语言推理)。
        # 此时 extract_code 会返回整个 response (无块时兜底), 把自然语言当代码执行会报
        # SyntaxError, 误导成"代码语法错"。改为明确标记"未输出代码块", 不执行残缺内容。
        # correct 仍为 False (诚实: 模型确实没产出可执行代码), 仅让错误信息更准确可区分。
        if response and response.strip() and not CODE_BLOCK_RE.search(response):
            return SampleResult(
                sample_id=sample.sample_id,
                response=response,
                extracted="",
                correct=False,
                score=0.0,
                error="未输出代码块: response 为自然语言推理 (思维链泄漏/被 max_tokens 截断, 未产出代码)",
            )
        if sample.meta.get("exec_mode") == "stdin":
            # stdin 式题: 对每 case 跑被测程序、喂 stdin、规范化比对 stdout。
            cases = _parse_lcb_stdin_cases(sample.test_code or "")
            passed, err = run_code_stdin(completion, cases)
            return SampleResult(
                sample_id=sample.sample_id,
                response=response,
                extracted=completion[:200] if completion else "",
                correct=passed,
                score=1.0 if passed else 0.0,
                error=err or None,
            )
        # 函数式题 (默认): completion + harness 拼一坨跑, 比对返回值。
        entry_point = sample.meta.get("entry_point", "")
        test_cases = _parse_lcb_tests(sample.test_code or "")

        passed = False
        err = ""
        if completion and entry_point and test_cases:
            harness = _build_v6_harness(entry_point, test_cases)
            passed, err = run_code_tests(completion, harness, entry_point=entry_point)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=completion[:200] if completion else "",
            correct=passed,
            score=1.0 if passed else 0.0,
            error=err or None,
        )


def _parse_lcb_stdin_cases(test_code: str) -> List[dict]:
    """把 stdin 题的 test_code (JSON 串) 解析为 [{"input":..., "output":...}, ...]。

    stdin 题的 test_code 是 public + private 合并后的 case 列表, 每个 case 含
    input/output/testtype 字段。run_code_stdin 只需 input/output。
    """
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
        cases.append({
            "input": str(tc.get("input", "")),
            "output": str(tc.get("output", "")),
        })
    return cases


def _build_v6_harness(entry_point: str, cases: List[dict]) -> str:
    """生成 v6 测试 harness: 实例化 Solution() 调用 entry_point 方法, 断言返回值 == expected。

    与父类 _build_lcb_harness 的差异: v6 题目是 class Solution 方法 (带 self),
    需 Solution().method(*args) 而非 method(*args)。用 json 序列化比较返回值。
    """
    import json as _json
    lines = ["import json as _json", ""]
    for i, c in enumerate(cases):
        args_repr = _json_safe_repr(c["args"])
        exp_repr = _json_safe_repr(c["expected"])
        lines.append(f"_args{i} = {args_repr}")
        lines.append(f"_exp{i} = {exp_repr}")
        lines.append(f"_got{i} = Solution().{entry_point}(*_args{i})")
        lines.append(
            "assert _json.dumps(_got%d, sort_keys=True) == _json.dumps(_exp%d, sort_keys=True), "
            "'case %d failed: got ' + repr(_got%d) + ' want ' + repr(_exp%d)" % (i, i, i, i, i)
        )
    return "\n".join(lines)


def _json_safe_repr(obj) -> str:
    """把 Python 对象转成可 eval 的 Python 字面量 (嵌入 harness 当源码执行)。

    必须用 repr 而非 json.dumps: json.dumps(True)→"true"、json.dumps(None)→"null",
    这些在 Python 里不是合法字面量, 写进 harness 会 NameError("name 'true' is not defined")。
    repr(True)→"True"、repr(None)→"None" 才是合法 Python 字面量。LCB-v6 函数式题里
    返回 bool 的题 (doesValidArrayExist/isFascinating 等, 444 题中 31 题) 全受此影响。
    """
    return repr(obj)
