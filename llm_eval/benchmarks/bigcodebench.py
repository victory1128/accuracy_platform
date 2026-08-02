"""BigCodeBench: 实用代码生成评测

1140 个任务, 每个含 instruct 说明 + 函数签名 (def task_func...) + unittest 测试。
prompt 已含 "You should write self-contained code starting with: ... def task_func(...)":
模型需补全完整可运行代码 (含 imports), 执行 test_code (unittest 类, 调 task_func) 验 pass@1。
比 HumanEval 难: 需调用标准库/工具, 代码更长。
"""
from __future__ import annotations

import os
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from ..scoring import extract_code
from ..scoring.code_exec import run_code_tests
from .base import Benchmark
from .registry import register
from ._data import data_path

BCB_HEADER = (
    "Complete the task below by writing a complete, self-contained Python function. "
    "Output only the code in a ```python``` block, including all needed imports and the "
    "function definition. The function must match the given signature exactly."
)


class _BigCodeBench(Benchmark):
    """BigCodeBench 基类: 复用 _eval_code, 但 prompt 风格独立 (instruct+签名)。"""

    META: BenchmarkMeta = None  # type: ignore[assignment]
    DATA_FILE: str = ""

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path(self.DATA_FILE)
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            sid = r.get("sample_id") or r.get("task_id") or ""
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
        return f"{BCB_HEADER}\n\n{sample.question}\n\n```python\n"

    def parse_params(self) -> dict:
        # 实用代码任务更长, 给足预算; 不设 stop (思维链模型会先思考)。
        return {"temperature": 0.0, "max_tokens": 4096, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        """BigCodeBench 的 test_code 是 unittest.TestCase 类, 需追加 runner 才会执行。

        (HumanEval/MBPP 的 test 是顶层裸 assert, 导入即执行; BigCodeBench 的断言
        在 TestCase 方法里, 不调 unittest.main() 就不会跑——会误判全部通过。)
        """
        completion = extract_code(response)
        passed = False
        err = ""
        if completion and sample.test_code:
            # 追加 unittest 运行器: 默认 exit=True, 测试失败时 sys.exit(非0) ->
            # 子进程 returncode != 0 -> run_code_tests 判为未通过。
            # (不能 exit=False: 那样失败只打印不退出, returncode 仍为0, 会误判通过。)
            harness = sample.test_code + (
                "\n\nif __name__ == '__main__':\n"
                "    import unittest\n"
                "    unittest.main(verbosity=0)\n"
            )
            passed, err = run_code_tests(completion, harness)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=completion[:200] if completion else "",
            correct=passed,
            score=1.0 if passed else 0.0,
            error=err or None,
        )


@register
class BigCodeBench(_BigCodeBench):
    DATA_FILE = "bigcodebench.jsonl"
    META = BenchmarkMeta(
        name="bigcodebench",
        display_name="BigCodeBench",
        stage=Stage.PRETRAIN,
        task_type=TaskType.CODE,
        description="实用代码生成, 1140任务, 需调用标准库/工具, pass@1",
        tags=["modern", "code", "hard"],
        num_fewshot=0,
        source="Zhuo et al. 2024 (BigCode)",
    )
