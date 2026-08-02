"""后训练阶段评测集

IFEval (指令遵循, 规则可验证) - 不需裁判
MT-Bench / AlpacaEval / Arena-Hard (LLM 裁判打分) - 需裁判模型
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from .base import Benchmark
from .registry import register
from ._data import data_path


# ===================== IFEval (规则) =====================

@register
class IFEval(Benchmark):
    META = BenchmarkMeta(
        name="ifeval",
        display_name="IFEval",
        stage=Stage.POSTTRAIN,
        task_type=TaskType.RULE,
        description="指令遵循评测, 可验证的格式约束(字数/JSON/标题等)",
        tags=["modern", "instruction_following"],
        num_fewshot=0,
        needs_judge=False,
        source="Zhou et al. 2023 (Google)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("ifeval.jsonl")
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
                    meta={"constraints": r.get("constraints", [])},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        return sample.question

    def parse_params(self) -> dict:
        # 思维链模型: reasoning 先消耗大量 token, 1024 预算常被思维链吃光导致
        # 正式答案被截断(finish_reason=length)或完全空输出, 规则检查必然失败、
        # 分数被严重低估。给足预算让思维链+答案都能完整输出。
        return {"temperature": 0.7, "max_tokens": 8192, "stop": None}


# ===================== Judge 类公共基类 =====================

class _JudgeBenchmark(Benchmark):
    """LLM 裁判打分评测集基类 (MT-Bench / AlpacaEval / Arena-Hard)"""

    META: BenchmarkMeta = None  # type: ignore[assignment]
    DATA_FILE: str = ""

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path(self.DATA_FILE)
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
                    meta={"reference": r.get("reference", ""), "turn": r.get("turn", 1)},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        return sample.question

    def parse_params(self) -> dict:
        # 对齐类用稍高温度, 充分发挥。思维链模型需更大 token 预算。
        return {"temperature": 0.7, "max_tokens": 4096, "stop": None}


@register
class MTBench(_JudgeBenchmark):
    DATA_FILE = "mt_bench.jsonl"
    META = BenchmarkMeta(
        name="mt_bench",
        display_name="MT-Bench",
        stage=Stage.POSTTRAIN,
        task_type=TaskType.JUDGE,
        description="多轮对话能力评测, GPT-4裁判1-10分",
        tags=["modern", "alignment", "multiturn"],
        num_fewshot=0,
        needs_judge=True,
        source="Zheng et al. 2023 (LMSYS)",
    )


@register
class AlpacaEval(_JudgeBenchmark):
    DATA_FILE = "alpaca_eval.jsonl"
    META = BenchmarkMeta(
        name="alpaca_eval",
        display_name="AlpacaEval 2.0",
        stage=Stage.POSTTRAIN,
        task_type=TaskType.JUDGE,
        description="指令遵循+回答质量, 裁判胜率(胜过参考模型的比例)",
        tags=["modern", "alignment"],
        num_fewshot=0,
        needs_judge=True,
        source="Li et al. 2023 (Stanford)",
    )


@register
class ArenaHard(_JudgeBenchmark):
    DATA_FILE = "arena_hard.jsonl"
    META = BenchmarkMeta(
        name="arena_hard",
        display_name="Arena-Hard",
        stage=Stage.POSTTRAIN,
        task_type=TaskType.JUDGE,
        description="高难度指令遵循, 裁判配对胜率(vs GPT-4基线)",
        tags=["modern", "alignment", "hard"],
        num_fewshot=0,
        needs_judge=True,
        source="Li et al. 2024 (LMSYS)",
    )
