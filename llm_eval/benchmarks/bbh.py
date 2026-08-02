"""BBH (BigBench Hard): 23 项难推理任务

各子任务 gold 格式异构:
- 选项题: gold='(B)', 题面里选项写成 (A)/(B)...
- 判断题: gold='True'/'False'/'Yes'/'No'
- 数值题: gold='24' / gold='8'
- 文本题: gold='syndrome therefrom'

评分策略: 先尝试选项字母抽取 (题面有 (A)..(E) 时), 否则做归一化精确匹配
(去标点/大小写/冠词/空白), 兼容 "False" vs "false"、"(B)" vs "B" 等。
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from ..scoring import extract_choice, normalize_answer
from .base import Benchmark
from .registry import register
from ._data import data_path

BBH_HEADER = (
    "Answer the following question. If there are options labeled (A), (B), (C), etc., "
    "answer with the letter. Otherwise give the exact answer. Think step by step first, "
    "then give your final answer on the last line after 'Answer:'."
)


@register
class BBH(Benchmark):
    META = BenchmarkMeta(
        name="bbh",
        display_name="BigBench-Hard (BBH)",
        stage=Stage.PRETRAIN,
        task_type=TaskType.GEN,
        description="23项难推理任务 (逻辑/数学/常识), 需要多步推理",
        tags=["modern", "reasoning", "hard"],
        num_fewshot=3,
        source="Suzgun et al. 2022",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("bbh.jsonl")
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
        return f"{BBH_HEADER}\n\n{sample.question}\nAnswer:"

    def parse_params(self) -> dict:
        # BBH 是难推理题, 思维链长, 2048 会截断, 给 8192。
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

        # 1) gold 是选项字母 (形如 "(B)" 或 "B") -> 抽字母
        # BBH 部分子任务选项多达 18 个 (reasoning_about_colored_objects: A-R)。
        # 从题面扫描实际出现的 (X) 选项字母作为合法集; 仅在题面确有选项时走字母抽取,
        # 否则降级到归一化匹配 (避免把 "True"/"False" 里的 T/F 误当选项)。
        gold_letter = gold.strip("()").upper()
        if len(gold_letter) == 1 and gold_letter.isalpha():
            import re as _re
            present = sorted(set(_re.findall(r"\(([A-Z])\)", sample.question)))
            if present:
                pred = extract_choice(response, present)
                if pred:
                    correct = pred == gold_letter

        # 2) 非选项题: 归一化精确匹配
        if not pred:
            pred = _bbh_extract_answer(response)
            correct = normalize_answer(pred) == normalize_answer(gold)

        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=pred,
            correct=correct,
            score=1.0 if correct else 0.0,
        )


def _bbh_extract_answer(response: str) -> str:
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
