"""答案抽取与归一化

从模型自由生成的输出里, 鲁棒地抽出:
- 选择题答案 (A/B/C/D)
- 数学题最终数值 (支持 \\boxed{}, "答案是 42", "The answer is 42" 等)
"""
from __future__ import annotations

import re
import string
import unicodedata
from typing import List, Optional

# Unicode 标点/符号 → ASCII 等价物 (用于 normalize_answer)。
# string.punctuation 只含 ASCII, 不覆盖 en-dash/em-dash/curly quote 等;
# gold 常含 '–' (en-dash) 而 pred 用 '-' (hyphen), 不归一会被误判不等。
_PUNCT_NORMALIZE = {
    0x2013: "-",   # en-dash
    0x2014: "-",   # em-dash
    0x2018: "'",   # left single quote
    0x2019: "'",   # right single quote
    0x201C: '"',   # left double quote
    0x201D: '"',   # right double quote
    0x00A0: " ",   # no-break space
    0x2010: "-",   # hyphen
    0x2011: "-",   # non-breaking hyphen
    0x2012: "-",   # figure dash
}

# 匹配 \boxed{...} (含嵌套大括号的简单处理)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
# "答案是 / answer is / 最终答案是 / 答案 50%" 后跟数字/表达式
ANSWER_IS_RE = re.compile(
    r"(?:答案是|最终答案(?:是)?|正确答案(?:是)?|答案(?:是)?|answer\s*is|final\s*answer\s*is|the\s*answer\s*is)\s*[:：]?\s*"
    r"([\-\$]?[\d,\.\/\s]+%?[A-Za-z%\^\[\]\(\)]*)",
    re.IGNORECASE,
)
# 最后一个数字 (兜底)
LAST_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*%?")

# 选项字母抽取: "答案选C" / "The answer is C" / "(C)" / "**C**" / 行首 "C."
# 支持 A-Z (覆盖 SuperGPQA 10选项 / BBH 多达18选项 A-R 等)。
# 注意: 正则范围放宽后, 必须靠 valid_choices 过滤掉非选项字母, 避免把
# "answer is True/Never" 里的 T/N 误当选项。MCQ 评测集调用时都会传 valid_choices。
# 关键: answer/choice/option 后须有词边界 (否则 "options" 里的 "option"+"s" 误匹配)。
CHOICE_RE_LIST = [
    re.compile(r"\\boxed\{\s*([A-Z])\s*\}", re.IGNORECASE),
    re.compile(r"(?:答案|选项)\s*(?:选|为|是)?\s*[:：]?\s*\(*([A-Z])\)*", re.IGNORECASE),
    # 中文推理常见: "选A" / "故选A" / "应选 A" / "因此选A" (无"答案/选项"前缀)
    # valid_choices 过滤保证只认合法选项字母, 误匹配风险低。
    re.compile(r"(?:故|应|因此|所以|则)?\s*选\s*[:：]?\s*\(*([A-Z])\)*", re.IGNORECASE),
    # "answer is X" / "answer: X" / "the answer is X" — 取最后一个匹配 (最终答案在末尾)
    re.compile(r"\banswer\b\s*(?:is|:)?\s*\(*([A-Z])\)*", re.IGNORECASE),
    re.compile(r"^\s*([A-Z])\s*$", re.MULTILINE),  # 单独一行只有一个字母 (推理完另起行给答案)
    re.compile(r"\b(?:choice|option)\b\s*(?:is|:)?\s*\(*([A-Z])\)*", re.IGNORECASE),
    re.compile(r"\b([A-Z])\b\s*[.。)]"),     # "A." "A)" 开头式
    re.compile(r"\(\s*([A-Z])\s*\)"),         # "(A)"
    re.compile(r"\*\*([A-Z])\*\*"),           # "**A**"
]


def normalize_answer(ans: str) -> str:
    """通用答案归一化: 去空白/标点/大小写, 用于精确匹配。

    在去 ASCII 标点 (string.punctuation) 基础上, 额外处理两类 Unicode 差异,
    否则 gold/pred 因字符编码不同被误判不等:
    1. 重音字母折叠为基本字母 (NFKD 分解后丢弃组合标记): café→cafe, naïve→naive,
       Zürich→zurich。SimpleQA 等 gold 含 é/ä/è 等, 模型常输出无重音版本。
    2. Unicode 破折号/引号归一化为 ASCII: en-dash – (U+2013)/em-dash — (U+2014)
       → hyphen -, curly quote ' (U+2019)→', 同样被后续标点清理。
       gold 'Urbana–Champaign' (en-dash) vs pred 'Urbana-Champaign' (hyphen)
       原先 en-dash 不在 string.punctuation 里被保留 → 不等。
    """
    if ans is None:
        return ""
    ans = str(ans).strip().lower()
    # 重音折叠: NFKD 分解, 丢弃组合标记 (Combining Diacritical Marks U+0300–U+036F)
    ans = unicodedata.normalize("NFKD", ans)
    ans = "".join(ch for ch in ans if not unicodedata.combining(ch))
    # Unicode 破折号/引号 → ASCII (随后由标点清理统一处理)
    ans = ans.translate(_PUNCT_NORMALIZE)
    # 去掉冠词与常见标点
    ans = re.sub(r"\b(a|an|the)\b", " ", ans)
    # 去除所有标点
    ans = "".join(ch for ch in ans if ch not in string.punctuation)
    # 合并空白
    ans = " ".join(ans.split())
    return ans


def extract_choice(text: str, valid_choices: Optional[List[str]] = None) -> str:
    """从文本抽取选择题答案 (返回大写字母, 如 'C'); 抽不到返回 ''。

    对每个正则取**最后一个**匹配 (模型推理中可能提到多个选项, 但最终答案在末尾)。
    """
    if not text:
        return ""
    valid = [c.upper() for c in valid_choices] if valid_choices else None
    for pat in CHOICE_RE_LIST:
        matches = list(pat.finditer(text))
        if not matches:
            continue
        # 从后往前找第一个落在 valid 里的匹配 (最终答案优先)
        for m in reversed(matches):
            cand = m.group(1).upper()
            if valid is None or cand in valid:
                return cand
    # 兜底: 文本只剩单个字母 (A-Z)
    stripped = text.strip()
    if len(stripped) == 1 and stripped.isalpha() and stripped.isupper():
        if valid is None or stripped in valid:
            return stripped
    return ""


def _clean_number_str(s: str) -> str:
    """轻度清理: 去美元符号/前后空格, 保留逗号(区间/列表里可能是分隔符)"""
    return s.strip().replace("$", "").strip()


def _normalize_number_str(s: str) -> str:
    """把纯数字字符串归一化: 去千分位逗号/美元符号/空格 (仅对纯数值用)"""
    s = s.strip().replace(",", "").replace("$", "").replace(" ", "")
    return s


def _is_plain_number(s: str) -> bool:
    """判断是否是单纯数值 (可含 负号/小数点/百分号/分数斜杠), 非区间/列表"""
    s = s.strip()
    if not s:
        return False
    # 含括号/方括号 -> 区间或列表, 不是纯数值
    if any(c in s for c in "[](){}"):
        return False
    return bool(re.fullmatch(r"[-+]?\d[\d,]*\.?\d*%?|\d+\s*/\s*\d+", s))


def extract_number(text: str) -> str:
    """从数学题输出中抽取最终答案 (字符串形式)

    优先级: \\boxed{} > '答案是X' 句式 > 文本最后一个数字
    \\boxed{} 的内容原样返回 (可能是区间 [2,5) 或表达式), 其它做数值清理。
    """
    if not text:
        return ""
    # 1. boxed —— 原样保留内容 (支持区间/列表/表达式)
    m = BOXED_RE.search(text)
    if m:
        return _clean_number_str(m.group(1))
    # 2. "答案是X" 句式
    m = ANSWER_IS_RE.search(text)
    if m:
        return _clean_number_str(m.group(1))
    # 3. 取所有数字, 返回最后一个 (数学题答案通常在末尾)
    nums = LAST_NUMBER_RE.findall(text)
    if nums:
        return _normalize_number_str(nums[-1])
    return ""


def numbers_equal(a: str, b: str) -> bool:
    """判断两个答案是否相等 (容忍数值形式差异: 0.5 vs 1/2 vs 50%; 区间则精确匹配)"""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return a == b
    # 非纯数值 (区间/列表/表达式): 归一化空白后精确匹配
    if not (_is_plain_number(a) and _is_plain_number(b)):
        return _normalize_ws(a) == _normalize_ws(b)
    a, b = _normalize_number_str(a), _normalize_number_str(b)
    # 百分比: "50%" 视为 0.5, 与小数比较时换算
    if a.endswith("%") or b.endswith("%"):
        fa, fb = _to_float_pct(a), _to_float_pct(b)
        if fa is not None and fb is not None:
            return abs(fa - fb) < 1e-6
        return a == b
    # 分数 a/b
    fa, fb = _to_float(a), _to_float(b)
    if fa is None or fb is None:
        return a == b
    # 容忍浮点误差
    if abs(fa - fb) < 1e-6:
        return True
    # 相对误差 (大数)
    if fb != 0 and abs(fa - fb) / max(abs(fb), 1e-9) < 1e-4:
        return True
    return False


def _normalize_ws(s: str) -> str:
    """归一化空白 (区间答案 [2, 5) 与 [2,5) 视为相等)"""
    s = " ".join(s.split())
    # 去掉标点(逗号/括号)周围的空格
    s = re.sub(r"\s*([,()\[\]{}])\s*", r"\1", s)
    return s


def _to_float(s: str) -> Optional[float]:
    s = s.replace("%", "").strip()
    if "/" in s:
        parts = s.split("/")
        try:
            num, den = float(parts[0]), float(parts[1])
            return num / den if den else None
        except (ValueError, IndexError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_float_pct(s: str) -> Optional[float]:
    """把 '50%' 换算成 0.5; '0.5' 保持 0.5 (用于百分数与分数比较)"""
    s = s.strip()
    if s.endswith("%"):
        v = _to_float(s[:-1])
        return v / 100.0 if v is not None else None
    return _to_float(s)
