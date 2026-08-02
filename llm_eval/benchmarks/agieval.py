"""AGIEval: 人类水平综合能力评测

多科目标准化考试集合 (高考/LSAT/SAT/AQuA/LogiQA 等), 中英混合。
数据特点: question 字段已自带题面+选项+"A: Among A through X, the answer is" 尾巴,
choices 是带 (A)/(B) 前缀的文本, gold 是单字母 (A-E)。因此 build_prompt 不重复
拼接选项, 直接复用题面并要求输出单个字母。
"""
from __future__ import annotations

import os
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, Stage, TaskType
from .base import Benchmark
from .registry import register
from ._data import data_path

AGIEVAL_HEADER_ZH = "请仔细阅读题目并思考, 然后从给定选项中选出正确答案, 只输出单个字母(如 A)。"
AGIEVAL_HEADER_EN = "Read the question carefully, think step by step, then answer with a single letter (e.g. A)."


@register
class AGIEval(Benchmark):
    META = BenchmarkMeta(
        name="agieval",
        display_name="AGIEval",
        stage=Stage.PRETRAIN,
        task_type=TaskType.MCQ,
        description="人类水平综合能力, 高考/LSAT/SAT/AQuA/LogiQA 等标准化考试",
        tags=["modern", "knowledge", "reasoning", "chinese"],
        num_fewshot=0,
        source="Liu et al. 2023 (Microsoft)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("agieval.jsonl")
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            sid = r.get("sample_id") or ""
            gold = (r.get("gold") or "").strip().upper()
            # gold 个别是数字下标 (sat-en), 转字母
            if gold.isdigit():
                gi = int(gold)
                gold = "ABCDEFGHIJ"[gi] if 0 <= gi < 10 else gold
            samples.append(
                Sample(
                    sample_id=sid,
                    prompt="",
                    question=r.get("question", ""),
                    choices=r.get("choices") or [],
                    gold=gold,
                    meta={"subject": r.get("subject", "")},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        # 题面已含选项与 "A: Among A through X, the answer is" 尾巴,
        # 只需加思考指令; 中文科目按 subject 含 -zh 判定。
        subject = sample.meta.get("subject", "")
        lang = "zh" if subject.endswith("-zh") or subject.startswith("gaokao") else "en"
        header = AGIEVAL_HEADER_ZH if lang == "zh" else AGIEVAL_HEADER_EN
        return f"{header}\n\n{sample.question}"

    def parse_params(self) -> dict:
        # 思维链模型先思考再出字母, 给足预算防截断。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}
