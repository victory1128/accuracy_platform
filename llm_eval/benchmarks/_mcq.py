"""MCQ 评测集共用工具: few-shot prompt 构造

标准 MCQ prompt 模板 (MMLU 风格):
    The following are multiple choice questions (with answers) about X.

    Question: ...
    A. ...
    B. ...
    C. ...
    D. ...
    Answer: A

    (few-shot 示例若干, 然后)
    Question: <test>
    A. ...
    Answer:
"""
from __future__ import annotations

from typing import List, Optional

from ..models import Sample

MCQ_HEADER_EN = "The following are multiple choice questions (with answers). Think step by step and then answer with a single letter (A/B/C/D)."
MCQ_HEADER_ZH = "以下是多项选择题(附答案)。请仔细思考, 然后用单个字母(A/B/C/D)作答。"


def format_mcq_question(question: str, choices: List[str], lang: str = "en") -> str:
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [question.strip(), ""]
    for L, c in zip(letters, choices):
        lines.append(f"{L}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def format_mcq_fewshot(
    shots: List[Sample],
    question: str,
    choices: List[str],
    lang: str = "en",
) -> str:
    header = MCQ_HEADER_ZH if lang == "zh" else MCQ_HEADER_EN
    parts = [header, ""]
    for s in shots:
        parts.append(format_mcq_question(s.question, s.choices or [], lang))
        parts.append((s.gold or "").upper())
        parts.append("")
    parts.append(format_mcq_question(question, choices, lang))
    return "\n".join(parts)
