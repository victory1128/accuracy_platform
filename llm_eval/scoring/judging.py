"""LLM 裁判打分 (MT-Bench / AlpacaEval / Arena-Hard 用)

用另一个(更强的)模型对被测模型的输出打分。支持:
- 单答案打分 (1~10)
- 配对比较 (A vs B, 返回 A赢/B赢/平局)

裁判 prompt 借鉴 MT-Bench / Arena-Hard 的单答案 grading 风格, 让裁判输出结构化的
JSON, 便于解析分数。
"""
from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from ..client import LLMClient

# MT-Bench 风格单答案评分 prompt
SINGLE_JUDGE_PROMPT = """你是一个严格的评分员, 请对下面助手针对用户问题的回答进行评分。

[用户问题]
{question}

[参考答案]
{reference}

[助手回答]
{answer}

评分标准 (1-10 分, 允许小数):
- 10: 完美, 全面准确, 表达优秀
- 8-9: 很好, 基本正确, 小瑕疵
- 6-7: 合格, 有部分错误或遗漏
- 4-5: 较差, 关键错误
- 1-3: 很差, 完全错误或答非所问

请只输出一行 JSON, 格式: {{"score": <1-10的数字>, "reason": "<简短理由>"}}
"""


def _parse_judge_json(text: str) -> Tuple[Optional[float], str]:
    """从裁判输出里解析 {score, reason}"""
    if not text:
        return None, ""
    # 找第一个 JSON 对象
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = obj.get("score")
            if isinstance(score, (int, float)):
                return float(score), str(obj.get("reason", ""))
        except json.JSONDecodeError:
            pass
    # 兜底: 找 "score": 8 这类
    m = re.search(r"score['\"\s:]*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if m:
        return float(m.group(1)), text[:120]
    return None, text[:120]


def judge_single(
    judge_client: LLMClient,
    question: str,
    answer: str,
    reference: str = "",
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> Tuple[Optional[float], str]:
    """裁判对单条回答打分, 返回 (1~10的分数, 理由)"""
    prompt = SINGLE_JUDGE_PROMPT.format(
        question=question, reference=reference or "(无)", answer=answer
    )
    text, _ = judge_client.chat(
        prompt, temperature=temperature, max_tokens=max_tokens
    )
    return _parse_judge_json(text)


def judge_pair(
    judge_client: LLMClient,
    question: str,
    answer_a: str,
    answer_b: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> str:
    """裁判对两个回答做配对比较, 返回 'A' / 'B' / 'tie'

    用于 Arena-Hard 风格。这里 A=被测模型, B=参考基线。
    """
    prompt = f"""你是一个严格的评审员, 比较两个助手对同一问题的回答。

[问题]
{question}

[助手A的回答]
{answer_a}

[助手B的回答]
{answer_b}

请判断哪个回答更好。只输出一个词: A / B / tie
"""
    text, _ = judge_client.chat(
        prompt, temperature=temperature, max_tokens=max_tokens
    )
    t = text.strip().lower()
    if t.startswith("a"):
        return "A"
    if t.startswith("b"):
        return "B"
    return "tie"
