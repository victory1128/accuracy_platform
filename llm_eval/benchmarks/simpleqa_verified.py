"""SimpleQA-Verified: OpenAI SimpleQA 的人工核验修订版 (google/simpleqa-verified, 1000题)

与现有 simpleqa (同数据源 google/simpleqa-verified) 的区别:
- 现有 simpleqa 用 Google 原版归一化精确匹配口径 (偏严) + 可选 LLM judge。
- 本集作为独立评测集, 数据相同但口径明确标注为"核验版", 复用 LLM 裁判三分类
  (A=正确 / B=错误 / C=未尝试), 无裁判时回退宽松匹配 + acceptable range。

注: 两者数据源相同, 分数可直接对比; 本集主要用于在报告里独立展示"核验版"口径。
gold 是简短文本 (人名/数字/日期/专名), 评分逻辑与 simpleqa.py 一致 (复用其函数)。
"""
from __future__ import annotations

import os
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from .base import Benchmark
from .registry import register
from ._data import data_path
# 复用 simpleqa 的评分函数 (clean_gold / extract / loose_match / grade_simpleqa)
from .simpleqa import (
    _clean_gold,
    _extract_short_answer,
    _loose_match,
    _in_acceptable_range,
    grade_simpleqa,
    SIMPLEQA_HEADER,
)


@register
class SimpleQAVerified(Benchmark):
    META = BenchmarkMeta(
        name="simpleqa_verified",
        display_name="SimpleQA-Verified",
        stage=Stage.POSTTRAIN,
        task_type=TaskType.GEN,
        description="SimpleQA 人工核验修订版, 1000题短答案事实性问答 (LLM裁判三分类)",
        tags=["modern", "knowledge", "factual", "verified"],
        num_fewshot=0,
        needs_judge=True,
        source="Haas et al. 2025 (Google, 核验版)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("simpleqa_verified.jsonl")
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
        # 思维链模型会先在 reasoning_content 思考很久, 需足够预算让思考+答案都输出。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        # 评分逻辑与 simpleqa.py 完全一致 (复用其函数), 仅数据源/META 不同。
        gold = _clean_gold(sample.gold or "")
        pred = _extract_short_answer(response)
        grade = None
        if judge_client is not None and pred:
            grade, reason = grade_simpleqa(judge_client, sample.question, pred, gold)
        if grade is None:
            correct = _loose_match(pred, gold)
            if not correct:
                correct = _in_acceptable_range(pred, sample.gold or "")
            grade = "A" if correct else ("C" if not pred else "B")
        else:
            correct = grade == "A"
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=pred,
            correct=correct,
            score=1.0 if correct else 0.0,
            analysis={"simpleqa_grade": grade} if grade else None,
        )
