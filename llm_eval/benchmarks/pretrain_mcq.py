"""预训练阶段 - MCQ 评测集

MMLU / MMLU-Pro / GPQA-Diamond / ARC / HellaSwag / WinoGrande / TruthfulQA
C-Eval / CMMLU (中文)

共享 MCQ 评测逻辑, 只是数据来源与 few-shot 数不同。
"""
from __future__ import annotations

import os
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, Stage, TaskType
from .base import Benchmark
from .registry import register
from ._mcq import format_mcq_fewshot, format_mcq_question, MCQ_HEADER_EN, MCQ_HEADER_ZH
from ._data import data_path


class _MCQBenchmark(Benchmark):
    """通用 MCQ 评测集基类, 子类只设 META + DATA_FILE + LANG + FEWSHOT"""

    META: BenchmarkMeta = None  # type: ignore[assignment]
    DATA_FILE: str = ""
    LANG: str = "en"            # en / zh
    FEWSHOT: int = 0
    # few-shot 示例 (人工写, 用于展示格式), 可为空
    FEWSHOT_EXAMPLES: List[Sample] = []

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path(self.DATA_FILE)
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            choices = r.get("choices") or []
            gold = (r.get("gold") or "").strip().upper()
            sid = r.get("sample_id") or r.get("id") or ""
            samples.append(
                Sample(
                    sample_id=sid,
                    prompt="",  # build_prompt 时拼
                    question=r.get("question", ""),
                    choices=choices,
                    gold=gold,
                    meta={"subject": r.get("subject", "")},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        if self.FEWSHOT and self.FEWSHOT_EXAMPLES:
            return format_mcq_fewshot(
                self.FEWSHOT_EXAMPLES[: self.FEWSHOT],
                sample.question,
                sample.choices or [],
                self.LANG,
            )
        header = MCQ_HEADER_ZH if self.LANG == "zh" else MCQ_HEADER_EN
        return f"{header}\n\n" + format_mcq_question(
            sample.question, sample.choices or [], self.LANG
        )

    def parse_params(self) -> dict:
        # MCQ 用贪心解码。注意: 思维链模型(如 GLM-5.2)会先在 reasoning_content
        # 里思考很久再输出选项字母, max_tokens 必须给足, 否则思维链吃光预算,
        # content 为空, 答案被截断成空 (实测 GPQA 在 2048 下 75% 被截断)。
        # 给 8192 让思考+选项都能完整输出; 模型早结束不会强制生成那么多, 不浪费。
        # stop 在思维链模型上会打断思考, 故去掉。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}


# -------------------- 各 MCQ 评测集 --------------------

@register
class MMLU(_MCQBenchmark):
    DATA_FILE = "mmlu.jsonl"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="mmlu",
        display_name="MMLU",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="大规模多任务语言理解, 57个学科的知识问答",
        tags=["modern", "knowledge", "classic"],
        num_fewshot=5,
        source="Hendrycks et al. 2021",
    )


@register
class MMLUPro(_MCQBenchmark):
    DATA_FILE = "mmlu_pro.jsonl"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="mmlu_pro",
        display_name="MMLU-Pro",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="MMLU增强版, 更难、更侧重推理, 10选项",
        tags=["modern", "knowledge", "reasoning"],
        num_fewshot=5,
        source="Wang et al. 2024",
    )


@register
class GPQA(_MCQBenchmark):
    DATA_FILE = "gpqa.jsonl"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="gpqa",
        display_name="GPQA-Diamond",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="研究生级科学问答, Google-proof, 极难",
        tags=["modern", "knowledge", "reasoning"],
        num_fewshot=0,
        source="Rein et al. 2023",
    )


@register
class CEval(_MCQBenchmark):
    DATA_FILE = "ceval.jsonl"
    LANG = "zh"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="ceval",
        display_name="C-Eval",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="中文大模型多学科评测, 52学科",
        tags=["modern", "knowledge", "chinese"],
        num_fewshot=5,
        source="Huang et al. 2023",
    )


@register
class CMMLU(_MCQBenchmark):
    DATA_FILE = "cmmlu.jsonl"
    LANG = "zh"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="cmmlu",
        display_name="CMMLU",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="中文多任务语言理解, 67学科",
        tags=["modern", "knowledge", "chinese"],
        num_fewshot=5,
        source="Li et al. 2023",
    )


@register
class ARC(_MCQBenchmark):
    DATA_FILE = "arc.jsonl"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="arc",
        display_name="ARC (Challenge)",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="小学科学推理题, 含需推理的难题",
        tags=["classic", "reasoning", "science"],
        num_fewshot=0,
        source="Clark et al. 2018",
    )


@register
class HellaSwag(_MCQBenchmark):
    DATA_FILE = "hellaswag.jsonl"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="hellaswag",
        display_name="HellaSwag",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="常识推理, 句子续写选择",
        tags=["classic", "reasoning", "commonsense"],
        num_fewshot=0,
        source="Zellers et al. 2019",
    )


@register
class WinoGrande(_MCQBenchmark):
    DATA_FILE = "winogrande.jsonl"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="winogrande",
        display_name="WinoGrande",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="共指消解, Winograd 模式挑战大规模版",
        tags=["classic", "reasoning", "commonsense"],
        num_fewshot=0,
        source="Sakaguchi et al. 2021",
    )


@register
class TruthfulQA(_MCQBenchmark):
    DATA_FILE = "truthfulqa.jsonl"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="truthfulqa",
        display_name="TruthfulQA (MC1)",
        # 测幻觉/事实性对齐, 属经典后训练指标 (与 GEN 版配套)。
        stage=Stage.POSTTRAIN,
        task_type=TaskType.MCQ,
        description="真实性问答, 检测模型是否会模仿人类误解 (幻觉/事实对齐, 后训练指标)",
        tags=["classic", "truthfulness", "factual"],
        num_fewshot=0,
        source="Lin et al. 2022",
    )


@register
class MMLURedux(_MCQBenchmark):
    """MMLU-Redux: MMLU 去噪人工重标版, 修正原 MMLU 的标注错误"""
    DATA_FILE = "mmlu_redux.jsonl"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="mmlu_redux",
        display_name="MMLU-Redux",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="MMLU 去噪重标版, 修正原始标注错误, 4选项",
        tags=["modern", "knowledge", "classic"],
        num_fewshot=5,
        source="Gema et al. 2024 (Edinburgh)",
    )


@register
class SuperGPQA(_MCQBenchmark):
    """SuperGPQA: 研究生级多学科选择题, 最高10选项 (A-J), 比 GPQA 更广更难"""
    DATA_FILE = "supergpqa.jsonl"
    FEWSHOT = 0
    META = BenchmarkMeta(
        name="supergpqa",
        display_name="SuperGPQA",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="研究生级多学科选择题, 265学科, 最高10选项, Google-proof",
        tags=["modern", "knowledge", "reasoning", "hard"],
        num_fewshot=0,
        source="M-A-P 2025",
    )
