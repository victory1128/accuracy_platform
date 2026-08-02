"""HLE (Humanity's Last Exam): 极难跨学科知识题

2500 题, 全部为开放题 (无选项)。gold 异构:
- 约 612 条是单字母 (A-J): 题面虽无选项, 但答案是一个字母;
- 其余为自由文本: 数学式/单词/数字/专名 (如 '\\mathbb{Z}', 'Rxf3, Rf1#', '18')。

评分: 单字母答案走选项抽取; 自由文本走归一化精确匹配。
注: HLE 官方用 LLM 裁判评分, 这里采用精确匹配作为可复现的保守基线——
对开放表述容忍弱, 实际分数会偏低, 报告中据此说明。
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

HLE_HEADER = (
    "This is an extremely difficult expert-level question. Think carefully and thoroughly, "
    "then give your final answer on the last line after 'Answer:'. "
    "If the answer is a single letter, output just that letter."
)


@register
class HLE(Benchmark):
    META = BenchmarkMeta(
        name="hle",
        display_name="HLE (Humanity's Last Exam)",
        stage=Stage.PRETRAIN,
        task_type=TaskType.GEN,
        description="人类终极考试, 极难跨学科专家级问题 (精确匹配基线)",
        tags=["modern", "knowledge", "reasoning", "hard"],
        num_fewshot=0,
        source="CAIS 2025",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("hle.jsonl")
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
                    meta={"subject": r.get("subject", "")},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        return f"{HLE_HEADER}\n\n{sample.question}\nAnswer:"

    def parse_params(self) -> dict:
        # 极难题需充分推理, 给足预算到 8192。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        gold = (sample.gold or "").strip()
        pred = ""
        correct = False

        # HLE 全部为开放题 (无选项 choices=[]), 即便 gold 是单字母 (T/V/K/N/R 等)
        # 也是短文本答案而非选择题字母。故统一走归一化精确匹配, 不用 extract_choice
        # (开放题里 extract_choice 会从自由文本里误抽任意字母, 造成假阳性)。
        pred = _hle_extract_answer(response)
        correct = normalize_answer(pred) == normalize_answer(gold)

        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=pred,
            correct=correct,
            score=1.0 if correct else 0.0,
        )


def _hle_extract_answer(response: str) -> str:
    """从响应抽取最终答案。

    优先取 'Answer:' / 'The answer is' / '答案是' 等引导词之后的最后一行;
    否则取响应最后一行。
    """
    if not response:
        return ""
    text = response.strip()
    low = text.lower()
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
        return lines[-1]
    return tail
