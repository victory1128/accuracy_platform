"""MRCR: 多轮长上下文检索 (Multi-Round Coreference Resolution, OpenAI)

在超长多轮对话 (可达百万字符) 中插入 "needle" 问题, 测模型能否从海量上下文
精确检索/操作特定信息 (如"把字符串X加到第2首歌前")。

数据: prompt=JSON消息列表(超长对话), answer=期望回答(通常含 random_string 前缀)。
评分: random_string 是否在输出中出现 + 归一化匹配 answer 关键部分。
注: 这是 needle-in-a-haystack 类评测, 答案较长, 主要看 random_string 前缀是否正确。
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from ..scoring import normalize_answer
from .base import Benchmark
from .registry import register
from ._data import data_path


@register
class MRCR(Benchmark):
    META = BenchmarkMeta(
        name="mrcr",
        display_name="MRCR",
        stage=Stage.PRETRAIN,
        task_type=TaskType.GEN,
        description="超长多轮上下文检索, needle-in-a-haystack, 百万字符级。⚠ 需 ≥512K token 上下文模型: 约 50% 样本 context 超 38万 token (最长 50万), GLM(38万上限)会大量 ContextWindowExceededError, 建议对 ≥512K 长上下文模型启用",
        tags=["modern", "long_context", "long_context_1m", "retrieval"],
        num_fewshot=0,
        source="Lee et al. 2024 (OpenAI)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("mrcr.jsonl")
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
                    question="",
                    gold=r.get("gold", ""),
                    meta={
                        "context": r.get("context", ""),
                        "random_string": "",
                    },
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        # context 已在下载时由 JSON 消息列表拼接成纯文本
        return sample.meta.get("context", "")

    def parse_params(self) -> dict:
        # 答案可长可短; 思维链模型需预算。context 长度由 prompt 承载。给 8192。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        gold = (sample.gold or "").strip()
        # MRCR 答案含 random_string 前缀: 主要是看该前缀是否出现在输出里
        # (从 gold 里抽取前导 random_string)。否则退化为归一化匹配。
        correct = False
        # gold 形如 "mWEa9DrPT3**Verse 1**...", 前缀是随机串
        rs = _extract_random_prefix(gold)
        if rs and len(rs) >= 6:
            correct = rs in response
        if not correct:
            correct = normalize_answer(response) == normalize_answer(gold)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=response[:200] if response else "",
            correct=correct,
            score=1.0 if correct else 0.0,
        )


def _extract_random_prefix(gold: str) -> str:
    """从 gold 答案提取前导 random_string (字母数字混合, 后跟正文)。
    形如 'mWEa9DrPT3**Verse 1**...' -> 'mWEa9DrPT3'
    """
    if not gold:
        return ""
    # 取开头连续的字母数字 (10位左右), 后面通常跟非字母数字(如 ** 或 空格)
    import re
    m = re.match(r"^([A-Za-z0-9]{6,20})(?![A-Za-z0-9])", gold)
    return m.group(1) if m else ""
