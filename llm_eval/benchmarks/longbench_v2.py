"""LongBench-V2: 长文本理解评测

503 题, 每题含超长 context (4k-100k+ 字符) + 问题 + 4选项 (A-D)。
难度分 short/medium/long/hard, 长度分 short/medium/long。
本质是带长上下文的多项选择, 评分=选项字母精确匹配。

注: context 很长, prompt 紧凑构造 (不重复说明), max_tokens 给小值 (只需输出字母)。
"""
from __future__ import annotations

import os
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, Stage, TaskType
from .base import Benchmark
from .registry import register
from ._data import data_path


@register
class LongBenchV2(Benchmark):
    META = BenchmarkMeta(
        name="longbench_v2",
        display_name="LongBench-V2",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="长文本理解, 4选项长上下文MCQ, 503题。⚠ 部分样本需 ≥1M token 上下文: context 中位数 19万 token 但长尾达 210万, 约 26% 超 38万 token, GLM(38万上限)对长尾样本会 ContextWindowExceededError",
        tags=["modern", "long_context", "long_context_1m", "reasoning"],
        num_fewshot=0,
        source="Bai et al. 2024 (THUDM)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("longbench_v2.jsonl")
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            sid = r.get("sample_id") or ""
            gold = (r.get("gold") or "").strip().upper()
            samples.append(
                Sample(
                    sample_id=sid,
                    prompt="",
                    question=r.get("question", ""),
                    choices=r.get("choices") or [],
                    gold=gold,
                    meta={
                        "context": r.get("context", ""),
                        "domain": r.get("subject", ""),
                        "difficulty": r.get("difficulty", ""),
                        "length": r.get("length", ""),
                    },
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        context = sample.meta.get("context", "")
        letters = [chr(ord("A") + i) for i in range(len(sample.choices or []))]
        opts = "\n".join(f"{L}. {c}" for L, c in zip(letters, sample.choices or []))
        # 紧凑构造: context + 问题 + 选项 + 要求输出单个字母
        return (
            f"{context}\n\n"
            f"Question: {sample.question}\n\n"
            f"{opts}\n\n"
            "Answer with a single letter (A/B/C/D):"
        )

    def parse_params(self) -> dict:
        # 只需输出字母, 但思维链模型会先思考; 给适中预算。
        # context 很长由 prompt 承载, 不影响 max_tokens。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}
