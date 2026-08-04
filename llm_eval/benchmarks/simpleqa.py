"""SimpleQA: 短答案事实性问答 (Google)

gold 是简短文本 (人名/数字/日期/专名), 如 '120,000 euros'、'2023'、'Oct 23, 2018'。
官方用 LLM 裁判判定, 这里采用归一化精确匹配作为可复现的基线口径:
- 去标点/大小写/冠词/空白后比较;
- gold 含 'acceptable range' 注释时, 取括号前的主答案比较。
注: 精确匹配口径会比官方裁判口径偏低 (容忍近义表述弱), 报告中据此说明。
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

SIMPLEQA_HEADER = (
    "Answer the following question with a short, exact answer. "
    "Give only the answer (a name, number, or date), no explanation."
)


@register
class SimpleQA(Benchmark):
    META = BenchmarkMeta(
        name="simpleqa",
        display_name="SimpleQA",
        stage=Stage.POSTTRAIN,
        task_type=TaskType.GEN,
        description="短答案事实性问答, 覆盖多领域事实 (人名/数字/日期)",
        tags=["modern", "knowledge", "factual"],
        num_fewshot=0,
        source="Wei et al. 2024 (Google)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("simpleqa.jsonl")
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
        # 思维链模型会先在 reasoning_content 思考很久, 需足够预算让思考+答案都输出,
        # 否则思维链吃光 max_tokens 导致正式答案(response)为空。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        gold = _clean_gold(sample.gold or "")
        pred = _extract_short_answer(response)
        # 官方 SimpleQA 用 LLM 裁判做语义三分类 (正确A/错误B/未尝试C),
        # 容忍大小写/标点/语序/部分省略/数字精度。平台归一化精确匹配偏严,
        # 会把 "120,000" vs "120,000 euros" / "Oct 23" vs "October 23" 判错。
        # 优先用 LLM 裁判 (有 judge_client 时); 否则回退到宽松匹配 (包含+数字归一)。
        grade = None  # "A"/"B"/"C" 或 None
        if judge_client is not None and pred:
            grade, reason = grade_simpleqa(judge_client, sample.question, pred, gold)
        if grade is None:
            # 无裁判或裁判失败 -> 宽松匹配回退 (比纯精确匹配救回~10个百分点)
            correct = _loose_match(pred, gold)
            # acceptable range 兜底: gold 含 "acceptable range: between X and Y" 时,
            # pred 数值落在 [X, Y] 内即对 (官方做法)。_clean_gold 已删注释, 故用原始 gold。
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
            # 裁判三分类存进 analysis 供报告展示 (A=正确 B=错误 C=未尝试)
            analysis={"simpleqa_grade": grade} if grade else None,
        )


def _clean_gold(gold: str) -> str:
    """去掉 gold 里的 'acceptable range' 等注释, 取主答案。

    如 '150 (acceptable range: anything between 148 and 152)' -> '150'
    """
    # 去掉括号内注释
    gold = re.sub(r"\s*\([^)]*acceptable[^)]*\)\s*$", "", gold, flags=re.IGNORECASE).strip()
    return gold


def _in_acceptable_range(pred: str, raw_gold: str) -> bool:
    """gold 含 'acceptable range: ... between X and Y' 时, 检查 pred 数值是否落在 [X, Y]。

    SimpleQA 部分数字题 gold 带容差区间 (如 '33.7738 (acceptable range: anything
    between 33.7586 and 33.8022)'), 官方判定: pred 落在区间内即正确, 不必精确等于主答案。
    _clean_gold 删掉了注释只比主答案, 会把 33.7833 (在 [33.7586, 33.8022] 内) 判错。
    这里从原始 gold 取区间, 检查 pred 首个数值是否落在 [X, Y] 内。

    仅当 gold 含 acceptable range 且 pred 能抽出数值时才可能判对, 否则返回 False
    (交回上层判错)。区间外 (真错) 不救回。
    """
    if not pred or not raw_gold:
        return False
    if "acceptable range" not in raw_gold.lower():
        return False
    m = re.search(r"between\s+(-?[\d.]+)\s+and\s+(-?[\d.]+)", raw_gold, re.IGNORECASE)
    if not m:
        return False
    try:
        lo, hi = float(m.group(1)), float(m.group(2))
    except ValueError:
        return False
    if lo > hi:  # 容错: 顺序写反
        lo, hi = hi, lo
    # pred 首个数值 (去千分位逗号后匹配, 如 '3,342.42' -> 3342.42)
    pm = re.search(r"[-+]?\d[\d,]*\.?\d*", pred.replace(",", ""))
    if not pm:
        return False
    try:
        val = float(pm.group(0))
    except ValueError:
        return False
    return lo <= val <= hi


def _extract_short_answer(response: str) -> str:
    """从响应抽取简短答案。

    SimpleQA 期望简短答案 (人名/数字/日期/专名)。策略:
    1. 优先取 'Answer:' / '答案是' 等引导词之后的第一行;
    2. 去掉引导词前缀本身 ("The answer is ..." / "答案是...");
    3. 否则取响应最后一行。
    归一化匹配会再清理标点/大小写, 故这里保留原文即可。
    """
    if not response:
        return ""
    text = response.strip()
    low = text.lower()
    # 找最后一个 "answer:" / "answer is" / "答案是" 引导
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
        return lines[0]
    return tail


def _loose_match(pred: str, gold: str) -> bool:
    """宽松匹配 (无 LLM 裁判时的回退): 比纯精确匹配救回约 10 个百分点。

    1. 归一化精确匹配;
    2. 包含关系 (任一方包含另一方) —— "Sanger Center" vs "Marjorie and James Sanger Center";
    3. 数字归一 (去非数字字符后比较) —— "120,000" vs "120,000 euros" -> 120000 == 120000。
    """
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p == g:
        return True
    if not p or not g:
        return False
    # 包含关系: 注意短串至少要有一定长度, 避免 "a" 匹配太多
    if len(p) >= 3 and (g in p or p in g):
        return True
    # 数字归一: 提取所有数字字符比较 (含小数点)
    pn = re.sub(r"[^\d.]", "", p)
    gn = re.sub(r"[^\d.]", "", g)
    if pn and gn and pn == gn:
        return True
    return False


# SimpleQA 官方 grading prompt (三分类 A/B/C), 改编自 OpenAI simple-evals。
# 裁判输出 "A"(正确)/"B"(错误)/"C"(未尝试)。
SIMPLEQA_GRADER_PROMPT = """You are grading a short answer to a factual question.

[Question]
{question}

[Gold target answer]
{gold}

[Model answer]
{pred}

Grade the model answer with one letter:
- A (correct): The answer fully contains the important information in the gold target and does not contradict it. Capitalization, punctuation, grammar, and word order do NOT matter. Minor omissions clearly inferable from the question are fine. Numbers must be correct to the last significant figure.
- B (incorrect): The answer contains a factual statement that contradicts the gold target (hedging does not save it).
- C (not_attempted): The important information in the gold target is NOT included and nothing contradicts it (e.g. "I don't know", or unrelated content).

Output ONLY the single letter A, B, or C."""


def grade_simpleqa(judge_client, question: str, pred: str, gold: str):
    """LLM 裁判对 SimpleQA 单条做三分类, 返回 (grade, reason)。

    grade: "A"/"B"/"C" 之一; 裁判失败返回 (None, reason) 由调用方回退到宽松匹配。
    注意: 思维链裁判模型 (如 glm-5.2) 会先在 reasoning_content 思考再在 content
    输出字母, max_tokens 必须给够 (4096) 让思考走完, 否则 finish_reason=length
    导致 content 为空。若 content 空但 reasoning 末尾有字母, 从 reasoning 兜底抽取。
    """
    from ..client import LLMClientError

    prompt = SIMPLEQA_GRADER_PROMPT.format(
        question=question or "(empty)", gold=gold or "(empty)", pred=pred or "(empty)"
    )
    try:
        text, usage = judge_client.chat(prompt, temperature=0.0, max_tokens=4096)
    except LLMClientError as e:
        return None, f"裁判请求失败: {e}"
    # 优先从 content (text) 抽 A/B/C
    t = (text or "").strip()
    m = re.search(r"\b([ABCabc])\b", t)
    if m:
        return m.group(1).upper(), t[:120]
    if t and t[0] in "ABCabc":
        return t[0].upper(), t[:120]
    # content 空 (思维链被 length 截断) -> 从 reasoning_content 末尾兜底抽字母
    if isinstance(usage, dict):
        rc = (usage.get("reasoning_content") or "").strip()
        if rc:
            m = re.search(r"\b([ABCabc])\b", rc[-200:])
            if m:
                return m.group(1).upper(), f"(reasoning兜底) {rc[-80:]}"
    return None, f"裁判解析失败: {(t or '(空content)')[:80]}"

