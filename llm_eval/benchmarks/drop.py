"""DROP: 阅读理解 + 离散推理 (Discrete Reasoning Over Paragraphs)

给一段 passage + question, 答案是文本 span (可能多个, 任一匹配即对)。
需在段落中做计数/排序/比较等离散推理 (如"谁得分最高"、"几次达阵")。
评分: 从模型输出抽取答案, 归一化后与任一 gold span 匹配。
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

DROP_HEADER = (
    "Read the passage and answer the question. The answer is usually a short span "
    "(a name, number, or phrase) from the passage. Give only the answer, no explanation.\n"
    "Answer:"
)


@register
class DROP(Benchmark):
    META = BenchmarkMeta(
        name="drop",
        display_name="DROP",
        stage=Stage.PRETRAIN,
        task_type=TaskType.GEN,
        description="阅读理解+离散推理, 需在段落中计数/排序/比较",
        tags=["modern", "reasoning", "reading_comprehension"],
        num_fewshot=3,
        source="Dua et al. 2019",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("drop.jsonl")
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            sid = r.get("sample_id") or ""
            gold = r.get("gold", [])
            # gold 是 spans 列表; 兼容单字符串
            if isinstance(gold, str):
                gold = [gold]
            samples.append(
                Sample(
                    sample_id=sid,
                    prompt="",
                    question=r.get("question", ""),
                    gold=gold,
                    meta={"passage": r.get("passage", "")},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        passage = sample.meta.get("passage", "")
        return f"Passage:\n{passage}\n\nQuestion: {sample.question}\n{DROP_HEADER}"

    def parse_params(self) -> dict:
        # 思维链模型会先推理, 256 token 不够思维链+答案, 会直接 length 截断导致全错。
        # 给足预算让思考+答案都完整输出。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        pred = _drop_extract_answer(response)
        golds = sample.gold if isinstance(sample.gold, list) else [sample.gold or ""]
        # 归一化后与任一 gold span 匹配 (数字也做数值比较)
        correct = any(_drop_match(pred, g) for g in golds if g)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=pred,
            correct=correct,
            score=1.0 if correct else 0.0,
        )


def _drop_extract_answer(response: str) -> str:
    """从响应抽取短答案: 取 'Answer:' 后第一行, 否则取最后一行。"""
    if not response:
        return ""
    text = response.strip()
    low = text.lower()
    for pat in (r"answer\s*:\s*", r"answer\s+is\s*"):
        m = list(re.finditer(pat, low))
        if m:
            text = text[m[-1].end():]
            break
    text = text.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else text


def _drop_match(pred: str, gold: str) -> bool:
    """DROP 答案匹配: 归一化精确匹配 + 数值相等容忍。"""
    if not pred or not gold:
        return False
    if normalize_answer(pred) == normalize_answer(gold):
        return True
    # 数值答案: "23" vs "23.0" / "twenty three"
    from ..scoring.extract import numbers_equal
    return numbers_equal(pred, gold)
