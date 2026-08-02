"""CorpusQA: 超长语料问答 (Tongyi-Zhiwen/CorpusQA)

在百万字符级语料 (1M tokens) 上做问答, 测模型超长上下文理解与检索。
格式: context (超长) + question + answer (短答案)。
评分: 归一化精确匹配 answer。
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


@register
class CorpusQA(Benchmark):
    META = BenchmarkMeta(
        name="corpusqa",
        display_name="CorpusQA",
        stage=Stage.PRETRAIN,
        task_type=TaskType.GEN,
        description="超长语料问答, 百万字符级上下文理解与检索。⚠ 需 ≥1M token 上下文模型: 全部 329 条 context 均 47万-215万 token, GLM(38万上限)等会全量 ContextWindowExceededError, 建议仅对百万级长上下文模型启用",
        tags=["modern", "long_context", "long_context_1m", "qa"],
        num_fewshot=0,
        source="Tongyi-Zhiwen 2025",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("corpusqa.jsonl")
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            sid = r.get("sample_id") or ""
            # gold 可能是 list (多个正确答案, 任一匹配即对), 用 || 拼成字符串保存。
            gold = r.get("gold", "")
            if isinstance(gold, list):
                gold = " || ".join(str(g).strip() for g in gold if str(g).strip())
            else:
                gold = str(gold).strip()
            samples.append(
                Sample(
                    sample_id=sid,
                    prompt="",
                    question=r.get("question", ""),
                    gold=gold,
                    meta={"context": r.get("context", "")},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        context = sample.meta.get("context", "")
        return (
            f"{context}\n\n"
            f"Based on the above text, answer the question with a short exact answer. "
            "Give only the answer, no explanation.\n"
            f"Question: {sample.question}\nAnswer:"
        )

    def parse_params(self) -> dict:
        # 思维链模型会先在 reasoning_content 推理(长上下文检索题尤其需充分思考),
        # 256 token 连思维链开头都装不下, 直接 finish_reason=length 截断, 永远没机会
        # 输出正式答案。给足预算让思考+答案都完整输出。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        pred = _corpusqa_extract(response)
        gold = (sample.gold or "").strip()
        # gold 可能是 "a || b || c" (多个正确答案, 任一匹配即对)。
        golds = [g.strip() for g in gold.split("||") if g.strip()]
        if not golds:
            golds = [""]
        pred_n = normalize_answer(pred)
        correct = any(pred_n == normalize_answer(g) for g in golds)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=pred,
            correct=correct,
            score=1.0 if correct else 0.0,
        )


def _corpusqa_extract(response: str) -> str:
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
