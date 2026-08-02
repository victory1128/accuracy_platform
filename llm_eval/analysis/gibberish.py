"""乱码 / 异常输出分析器

检测大模型输出中常见的"坏输出"信号, 为每条响应给出结构化分析:
1. 编码异常 (mojibake): 出现 UTF-8 被错误解码的典型字符 (ï¿½, Ã¢ 等)
2. 控制字符 / 非打印字符
3. 重复退化 (repetition degeneration): n-gram 重复率过高 (模型卡死复读)
4. 字符集失衡: 非预期脚本比例过高 (如纯回答里大量西里尔/泰文等)
5. 语言一致性: 中英混杂比例 (对中文题却全英文等)
6. 截断: 疑似因 max_tokens 截断 (无终止标点 / 末尾不完整)
7. 空输出 / 过短输出

输出一个 dict, 含 is_suspicious 标志 + 各维度细节 + 综合诊断字符串。
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# UTF-8 被当作 latin-1/cp1252 解码的典型 mojibake 字符序列
MOJIBAKE_PATTERNS = [
    re.compile(r"ï¿½"),
    re.compile(r"[ÃÂÄ][-¿]"),       # Ã¢ Â€ 等
    re.compile(r"â€[-¿]?"),
    re.compile(r"[�]"),                   # U+FFFD 替换符
    re.compile(r"Ã[-¿]"),
]

# 控制字符 (除常见空白 \t\n\r 外)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 终止性标点 (用于判断是否被截断)
TERMINAL_PUNCT = set("。.!！?？;；…\n")


def _script_of(ch: str) -> str:
    """返回字符所属脚本大类"""
    cp = ord(ch)
    if ch.isspace():
        return "space"
    # CJK 统一汉字 + 扩展
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "han"
    if 0x3040 <= cp <= 0x30FF:  # 平假名+片假名
        return "kana"
    if 0xAC00 <= cp <= 0xD7AF:  # 韩文音节
        return "hangul"
    if "A" <= ch <= "Z" or "a" <= ch <= "z":
        return "latin"
    if "0" <= ch <= "9":
        return "digit"
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return "punct"
    # 数学/排版符号 (Sm 数学符号 / Sc 货币符号 / Sk 修饰符号 / 反斜杠):
    # $ 是 LaTeX 数学定界符, = + - < > ^ 等是数学运算符, \ 是 LaTeX 命令前缀。
    # 这些在数学/代码输出里正常大量出现, 不应算"异常脚本"误报乱码。
    if cat.startswith("S") or ch == "\\":
        return "symbol"
    # 其它脚本: 西里尔/阿拉伯/泰文等
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0x0600 <= cp <= 0x06FF:
        return "arabic"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    return "other"


def _max_ngram_repetition(text: str, n: int = 3) -> Tuple[float, str]:
    """计算 n-gram 最大重复占比。

    返回 (重复率, 重复得最多的 ngram)。
    repetition degeneration 的典型表现: 同一个 token/ngram 占满整段输出。
    """
    tokens = text.split()
    if len(tokens) < n * 4:
        return 0.0, ""
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0, ""
    counter = Counter(grams)
    top_gram, top_count = counter.most_common(1)[0]
    repeat_ratio = (top_count * n) / len(tokens)
    return repeat_ratio, top_gram


def _char_repetition_ratio(text: str) -> float:
    """单字符连续重复 (如 aaaaaaa) 的最大占比"""
    if not text:
        return 0.0
    max_run = run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1] and not text[i].isspace():
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run / len(text)


def analyze_gibberish(text: str, expected_lang: str = "auto") -> Dict[str, Any]:
    """分析单条模型输出。

    Args:
        text: 模型输出文本
        expected_lang: 期望语言 "zh" / "en" / "auto"

    Returns 结构化分析 dict。
    """
    if text is None:
        text = ""
    length = len(text)
    result: Dict[str, Any] = {
        "length": length,
        "is_empty": length == 0,
        "is_short": 0 < length < 5,
        "mojibake": False,
        "mojibake_hits": 0,
        "control_chars": 0,
        "char_repetition_ratio": 0.0,
        "ngram_repetition_ratio": 0.0,
        "top_repeated_ngram": "",
        "script_distribution": {},
        "lang_match": True,
        "truncated": False,
        "diagnoses": [],          # 乱码/质量诊断 (mojibake/重复/控制字符/异常脚本)
        "abnormal_notes": [],     # 非乱码的输出异常 (空输出/截断), 单独计, 不进乱码率
        "lang_notes": [],         # 弱信号提示 (语言不一致/高度重复等), 不计入乱码率
        "is_suspicious": False,   # 仅乱码质量问题 (不含空输出/截断)
        "is_abnormal": False,     # 输出异常 (空输出/截断/思维链吃光), 与乱码分开
    }

    if length == 0:
        result["abnormal_notes"].append("空输出(empty)")
        result["is_abnormal"] = True
        return result

    # 1. mojibake
    moji_hits = 0
    for pat in MOJIBAKE_PATTERNS:
        moji_hits += len(pat.findall(text))
    if moji_hits > 0:
        result["mojibake"] = True
        result["mojibake_hits"] = moji_hits
        result["diagnoses"].append(f"编码乱码(mojibake) ×{moji_hits}")

    # 2. 控制字符
    ctrl = len(CONTROL_RE.findall(text))
    result["control_chars"] = ctrl
    if ctrl > 0:
        result["diagnoses"].append(f"含控制字符 ×{ctrl}")

    # 3. n-gram 重复 (repetition degeneration)
    rep_ratio, top_gram = _max_ngram_repetition(text, n=3)
    result["ngram_repetition_ratio"] = round(rep_ratio, 3)
    result["top_repeated_ngram"] = top_gram[:40]
    if rep_ratio >= 0.5:
        # ≥50%: 确定的重复退化 (模型卡死复读), 计入乱码
        result["diagnoses"].append(f"重复退化(ngram≥50%): “{top_gram[:20]}…”")
    elif rep_ratio >= 0.3:
        # 30%-50%: 弱信号。数学叙述里单位/短语(如 "\text{ mile}")反复会到这个区间,
        # 但并非乱码; 降级为 lang_notes 弱提示, 不计入乱码率 (is_suspicious)。
        result["lang_notes"].append(f"高度重复(ngram≥30%): “{top_gram[:20]}…”")

    # 单字符连续重复
    char_rep = _char_repetition_ratio(text)
    result["char_repetition_ratio"] = round(char_rep, 3)
    if char_rep >= 0.3 and length > 20:
        result["diagnoses"].append("单字符连续重复过高")

    # 4. 脚本分布
    scripts = Counter(_script_of(ch) for ch in text if not ch.isspace())
    total = sum(scripts.values()) or 1
    dist = {k: round(v / total, 3) for k, v in scripts.most_common()}
    result["script_distribution"] = dist

    # 5. 语言一致性
    # 注意: 仅对"有实质内容"的回答判断语言。MCQ 的单字母答案(A/B/C/D)、
    # 数值答案(42)等不构成语言信号, 不应误报。
    # 语言不一致(题目中文回答英文/反之)属于"风格偏好"而非"乱码",
    # 记录到 lang_notes 但不计入 suspicious (乱码率只反映真正的输出质量问题)。
    # lang_notes 也承载弱信号提示 (如 ngram 30%-50% 的高度重复), 不计入乱码率。
    han = dist.get("han", 0.0)
    latin = dist.get("latin", 0.0)
    substantive = sum(scripts.values()) >= 8
    # lang_notes 已在 n-gram 弱提示处 setdefault, 这里不覆盖
    # expected_lang="any" 表示题目明确要求某种语言 (如 IFEval 的 response_language 约束),
    # 此时模型用该语言回答是正确的, 不应报"语言不匹配"或"异常脚本占比过高"。
    if expected_lang != "any" and substantive:
        if expected_lang == "zh" and han < 0.05 and latin > 0.5:
            result["lang_match"] = False
            result["lang_notes"].append("期望中文但输出几乎全英文")
        elif expected_lang == "en" and latin < 0.05 and han > 0.5:
            result["lang_match"] = False
            result["lang_notes"].append("期望英文但输出几乎全中文")

    # 异常脚本占高 (非中英日韩的脚本突然大量出现)
    # expected_lang="any" 时跳过 (题目要求特定语言, 非预期脚本是正常的)
    if expected_lang != "any":
        exotic = dist.get("cyrillic", 0) + dist.get("arabic", 0) + dist.get("thai", 0) + dist.get("other", 0)
        if exotic >= 0.2:
            result["diagnoses"].append(f"异常脚本占比过高({exotic:.0%})")

    # 6. 截断判断 (保守: 仅在输出很长且末尾明显不完整时才报)
    # 正常回答常以数字/字母结尾 (如数学题答案), 不应误报。
    # 只有当长度接近典型 max_tokens 且末尾既无终止标点也非完整括号/引号闭合时才判为疑似截断。
    if length > 1500:
        last = text.rstrip()[-1:] if text.rstrip() else ""
        # 末尾是完整闭合 (括号/引号/代码块) 或终止标点 -> 不算截断
        if last and last not in TERMINAL_PUNCT and last not in ")]}>\"'`":
            result["truncated"] = True
            result["abnormal_notes"].append("疑似被截断(长度近上限且末尾无终止标点)")
            result["is_abnormal"] = True

    # is_suspicious 只反映乱码质量问题 (mojibake/重复/控制字符/异常脚本);
    # 空输出/截断属"输出异常"但非乱码, 归入 is_abnormal, 不进乱码率。
    result["is_suspicious"] = len(result["diagnoses"]) > 0
    return result


def summarize_gibberish(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对一个评测集所有样本的乱码分析做汇总"""
    n = len(results) or 1
    suspicious = sum(1 for r in results if r and r.get("is_suspicious"))
    abnormal = sum(1 for r in results if r and r.get("is_abnormal"))
    empty = sum(1 for r in results if r and r.get("is_empty"))
    mojibake = sum(1 for r in results if r and r.get("mojibake"))
    repetition = sum(1 for r in results if r and (r.get("ngram_repetition_ratio", 0) >= 0.3))
    truncated = sum(1 for r in results if r and r.get("truncated"))
    lang_mismatch = sum(1 for r in results if r and not r.get("lang_match", True))

    # 收集所有诊断文本 (真正的乱码/质量异常), 统计类型频次
    diag_counter: Counter = Counter()
    # 非乱码的输出异常 (空输出/截断) 单独统计, 不混入乱码诊断
    abn_counter: Counter = Counter()
    # 语言风格不一致单独统计 (不计入乱码诊断)
    lang_counter: Counter = Counter()
    for r in results:
        if not r:
            continue
        for d in r.get("diagnoses", []):
            key = re.split(r"[（(]", d)[0].strip()
            diag_counter[key] += 1
        for d in r.get("abnormal_notes", []):
            key = re.split(r"[（(]", d)[0].strip()
            abn_counter[key] += 1
        for d in r.get("lang_notes", []):
            lang_counter[d] += 1

    return {
        "total_samples": len(results),
        "suspicious_count": suspicious,
        "suspicious_rate": round(suspicious / n, 4),
        "abnormal_count": abnormal,             # 输出异常 (空输出/截断), 非乱码
        "abnormal_rate": round(abnormal / n, 4),
        "empty_count": empty,
        "mojibake_count": mojibake,
        "repetition_count": repetition,
        "truncated_count": truncated,
        "lang_mismatch_count": lang_mismatch,
        "lang_mismatch_notes": dict(lang_counter.most_common(10)),
        "diagnosis_counts": dict(diag_counter.most_common(10)),
        "abnormal_counts": dict(abn_counter.most_common(10)),
        "overall_grade": _grade(suspicious / n),
    }


def _grade(rate: float) -> str:
    """根据可疑率给一个乱码健康度等级"""
    if rate == 0:
        return "A (无异常)"
    if rate < 0.05:
        return "B (基本正常)"
    if rate < 0.15:
        return "C (少量异常)"
    if rate < 0.30:
        return "D (异常较多)"
    return "F (异常严重)"
