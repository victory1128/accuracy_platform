"""预训练阶段 - 数学/生成 评测集

GSM8K / MATH-500 / AIME
模型自由生成解题过程, 评分=从输出抽取最终数字与 gold 比较。
"""
from __future__ import annotations

import os
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, Stage, TaskType
from .base import Benchmark
from .registry import register
from ._data import data_path

GEN_HEADER_EN = "Solve the following math problem step by step. Put your final numerical answer after 'The answer is' or in \\boxed{}."
GEN_HEADER_ZH = "请逐步解决下面的数学题。把最终数值答案写在 \"答案是\" 之后, 或用 \\boxed{} 包裹。"


class _MathBenchmark(Benchmark):
    META: BenchmarkMeta = None  # type: ignore[assignment]
    DATA_FILE: str = ""

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path(self.DATA_FILE)
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            sid = r.get("sample_id") or r.get("id") or ""
            samples.append(
                Sample(
                    sample_id=sid,
                    prompt="",
                    question=r.get("question", ""),
                    gold=str(r.get("gold", "")).strip(),
                    meta={"level": r.get("level", "")},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        return f"{GEN_HEADER_ZH}\n\n题目:\n{sample.question}\n\n解答:"

    def parse_params(self) -> dict:
        # 数学题需要推理。思维链模型会大量消耗 reasoning token, 给足到 8192。
        # (实测 AIME 竞赛题在 4096 下 78% 被截断, 思维链吃光预算 content 为空。)
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}


@register
class GSM8K(_MathBenchmark):
    DATA_FILE = "gsm8k.jsonl"
    META = BenchmarkMeta(
        name="gsm8k",
        display_name="GSM8K",
        stage=Stage.PRETRAIN,
        task_type=TaskType.GEN,
        description="小学数学应用题, 多步推理",
        tags=["modern", "math", "reasoning", "classic"],
        num_fewshot=5,
        source="Cobbe et al. 2021",
    )


@register
class MATH500(_MathBenchmark):
    DATA_FILE = "math500.jsonl"
    META = BenchmarkMeta(
        name="math500",
        display_name="MATH-500",
        stage=Stage.PRETRAIN,
        task_type=TaskType.GEN,
        description="竞赛数学题, MATH 测试集500题子集",
        tags=["modern", "math", "reasoning"],
        num_fewshot=4,
        source="Lightman et al. 2023",
    )


@register
class AIME(_MathBenchmark):
    DATA_FILE = "aime.jsonl"
    META = BenchmarkMeta(
        name="aime",
        display_name="AIME",
        stage=Stage.PRETRAIN,
        task_type=TaskType.GEN,
        description="美国数学邀请赛, 高难度竞赛数学(答案0-999)",
        tags=["modern", "math", "reasoning", "hard"],
        num_fewshot=0,
        source="MAA",
    )

    def parse_params(self) -> dict:
        # AIME 是竞赛级难题, 思维链推理极长。max_tokens 演进 (实测 60 题):
        #   8192  -> 68% 截断, 56.7% 分
        #   16384 -> 62% 仍截断, 56.7% 分 (大量题思维链 >16384)
        #   65536 -> 45% 截断, 65.0% 分 (comp_tok 中位数31396/max65536)
        #           仍有 27/60 在 65536 下被截断 —— 思维链 >65536 token。
        #   131072-> 端点实测 HTTP 200 (上限≥131072), 覆盖剩余极端长推理题。
        # 其它数学集 (gsm8k/math500) 仍用基类 8192, 思维链没那么长。
        return {"temperature": 0.0, "max_tokens": 131072, "stop": None}


@register
class HMMTFeb2025(_MathBenchmark):
    """HMMT 2025 February (Harvard-MIT Mathematics Tournament).

    MathArena 维护的竞赛数学评测集, 30 题, 答案为最终数值/表达式。
    与 AIME 同属高难度竞赛数学, 思维链推理长, 用大 max_tokens。
    数据源: MathArena/hmmt_feb_2025 (HF), 字段 problem/answer。
    """
    DATA_FILE = "hmmt_feb_2025.jsonl"
    META = BenchmarkMeta(
        name="hmmt_feb_2025",
        display_name="HMMT-Feb-2025",
        stage=Stage.PRETRAIN,
        task_type=TaskType.GEN,
        description="哈佛-MIT 数学锦标赛 2025年2月赛, 30题高难度竞赛数学",
        tags=["modern", "math", "reasoning", "hard", "competition"],
        num_fewshot=0,
        source="MathArena 2025 (HMMT)",
    )

    def parse_params(self) -> dict:
        # 竞赛级难题, 思维链推理极长 (同 AIME), 给足 max_tokens。
        return {"temperature": 0.0, "max_tokens": 131072, "stop": None}


@register
class IMOAnswerBench(_MathBenchmark):
    """IMO-AnswerBench: 国际数学奥林匹克短答案评测。

    Google DeepMind 发布, 400 题, 答案为可验证短答案 (数字/表达式)。
    涵盖 Algebra/Combinatorics/Geometry/Number Theory 四大类。
    数据源: OpenEvals/IMO-AnswerBench (HF), 字段 Problem/Short Answer。
    """
    DATA_FILE = "imo_answerbench.jsonl"
    META = BenchmarkMeta(
        name="imo_answerbench",
        display_name="IMO-AnswerBench",
        stage=Stage.PRETRAIN,
        task_type=TaskType.GEN,
        description="国际数学奥林匹克短答案评测, 400题 (代数/组合/几何/数论)",
        tags=["modern", "math", "reasoning", "hard", "competition"],
        num_fewshot=0,
        source="Google DeepMind 2025 (IMO-Bench)",
    )

    def parse_params(self) -> dict:
        # 奥林匹克级难题, 思维链推理长 (同 AIME), 给足 max_tokens。
        return {"temperature": 0.0, "max_tokens": 131072, "stop": None}
