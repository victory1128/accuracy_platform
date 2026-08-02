"""DS-1000: 数据科学代码生成评测

1000 题, 覆盖 Pandas / NumPy / Matplotlib / SciPy / Scikit-learn。
每题 code_context 内含 test_execution(solution) 函数: 把模型代码(solution 字符串)
插入 exec_context 的 [insert] 位置执行, 用 exec_test 比对结果 (assert_frame_equal 等)。

评分: 提取模型 ```python 代码块, 拼成 `code_context + 调用 test_execution(模型代码)` ,
在沙箱执行 (需 llm-eval-sandbox 镜像, 含数据科学库)。
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

DS_HEADER = (
    "Solve the data science problem below using the relevant library "
    "(pandas/numpy/matplotlib/scipy/sklearn). Output only the Python code that completes "
    "the task in a ```python``` block. The code should define the function or produce the "
    "expected result so it can be inserted and executed."
)


@register
class DS1000(Benchmark):
    META = BenchmarkMeta(
        name="ds1000",
        display_name="DS-1000",
        stage=Stage.PRETRAIN,
        task_type=TaskType.CODE,
        description="数据科学代码生成, pandas/numpy/matplotlib, 1000题",
        tags=["modern", "code", "datascience"],
        num_fewshot=0,
        source="Lai et al. 2022",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("ds1000.jsonl")
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
                    gold=r.get("reference_code", ""),
                    test_code=r.get("code_context", ""),
                    meta={"library": r.get("library", "")},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        return f"{DS_HEADER}\n\n{sample.question}\n\n```python\n"

    def parse_params(self) -> dict:
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        completion = extract_code(response)
        passed = False
        err = ""
        if completion and sample.test_code and "test_execution" in sample.test_code:
            # code_context 自带 test_execution(solution); 把模型代码作为字符串传入。
            # 设环境变量让 matplotlib/seaborn 缓存写到可写 /tmp (docker 只读根)。
            sol_repr = _py_repr_str(completion)
            preamble = (
                "import os\n"
                "os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'\n"
                "os.environ['HOME'] = '/tmp'\n"
                "os.makedirs('/tmp/matplotlib', exist_ok=True)\n"
            )
            harness = preamble + sample.test_code + f"\n\ntest_execution({sol_repr})\n"
            passed, err = run_code_tests("", harness)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=completion[:200] if completion else "",
            correct=passed,
            score=1.0 if passed else 0.0,
            error=err or None,
        )


def _py_repr_str(s: str) -> str:
    """把模型代码转成 Python 字符串字面量 (用于 test_execution(代码字符串))。"""
    import json
    return json.dumps(s, ensure_ascii=False)
