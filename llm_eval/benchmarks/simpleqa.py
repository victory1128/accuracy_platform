"""SimpleQA: 短答案事实性问答 (Google)

gold 是简短文本 (人名/数字/日期/专名), 如 '120,000 euros'、'2023'、'Oct 23, 2018'。
官方用 LLM 裁判判定, 这里采用归一化精确匹配作为可复现的基线口径:
- 去标点/大小写/冠词/空白后比较;
- gold 含 'acceptable range' 注释时, 取括号前的主答案比较。
注: 精确匹配口径会比官方裁判口径偏低 (容忍近义表述弱), 报告中据此说明。
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from ..scoring import normalize_answer
from .base import Benchmark
from .registry import register
from ._data import data_path

SIMPLEQA_HEADER = (
    "Answer the following question with a short, exact answer. "
    "Give only the answer (a name, number, or date), no explanation."
)


@register
class SimpleQA(Benchmark):
    META = BenchmarkMeta(
        name="simpleqa",
        display_name="SimpleQA",
        stage=Stage.POSTTRAIN,
        task_type=TaskType.GEN,
        description="短答案事实性问答, 覆盖多领域事实 (人名/数字/日期)",
        tags=["modern", "knowledge", "factual"],
        num_fewshot=0,
        source="Wei et al. 2024 (Google)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("simpleqa.jsonl")
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
                    question=r.get("question", ""),
                    gold=str(r.get("gold", "")).strip(),
                    meta={"topic": r.get("topic", "")},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        return f"{SIMPLEQA_HEADER}\n\nQuestion: {sample.question}\nAnswer:"

    def parse_params(self) -> dict:
        # 思维链模型会先在 reasoning_content 思考很久, 需足够预算让思考+答案都输出,
        # 否则思维链吃光 max_tokens 导致正式答案(response)为空。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        gold = _clean_gold(sample.gold or "")
        pred = _extract_short_answer(response)
        correct = normalize_answer(pred) == normalize_answer(gold)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=pred,
            correct=correct,
            score=1.0 if correct else 0.0,
        )


def _clean_gold(gold: str) -> str:
    """去掉 gold 里的 'acceptable range' 等注释, 取主答案。

    如 '150 (acceptable range: anything between 148 and 152)' -> '150'
    """
    # 去掉括号内注释
    gold = re.sub(r"\s*\([^)]*acceptable[^)]*\)\s*$", "", gold, flags=re.IGNORECASE).strip()
    return gold


def _extract_short_answer(response: str) -> str:
    """从响应抽取简短答案。

    SimpleQA 期望简短答案 (人名/数字/日期/专名)。策略:
    1. 优先取 'Answer:' / '答案是' 等引导词之后的第一行;
    2. 去掉引导词前缀本身 ("The answer is ..." / "答案是...");
    3. 否则取响应最后一行。
    归一化匹配会再清理标点/大小写, 故这里保留原文即可。
    """
    if not response:
        return ""
    text = response.strip()
    low = text.lower()
    # 找最后一个 "answer:" / "answer is" / "答案是" 引导
    tail = text
    for pat in (r"answer\s*:\s*", r"answer\s+is\s*", r"final\s+answer\s*:\s*",
                r"答案是?\s*[:：]?\s*", r"最终答案(?:是)?\s*[:：]?\s*"):
        m = list(re.finditer(pat, low))
        if m:
            tail = text[m[-1].end():]
            break
    tail = tail.strip()
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if lines:
        return lines[0]
    return tail
