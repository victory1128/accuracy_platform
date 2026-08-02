#!/usr/bin/env python
"""下载完整评测数据集 -> 转成平台 jsonl 格式

用法:
  python scripts/download_datasets.py                # 下载全部
  python scripts/download_datasets.py mmlu gsm8k     # 只下指定数据集
  python scripts/download_datasets.py --list         # 列出可用数据集

输出到 llm_eval/benchmarks/data/<name>.jsonl, 覆盖现有 smoke-test 样本。
每个数据集独立下载, 单个失败不影响其他。

注意:
- 需先 pip install datasets huggingface_hub socksio (走代理时)
- GPQA 是 gated 数据集, 需先在 HF 页面同意条款并配 HF_TOKEN 环境变量
- HellaSwag/WinoGrande 用 validation split (test 无 label)
- LLM 裁判类 (mt_bench/arena_hard) 从 GitHub raw 下载
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any, Dict, List, Optional

# 平台 data 目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "llm_eval", "benchmarks", "data")
DATA_DIR = os.path.abspath(DATA_DIR)

# 是否走 HF 镜像 (国内直连慢时可设 HF_ENDPOINT=https://hf-mirror.com)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")


def _save_jsonl(name: str, rows: List[Dict[str, Any]]) -> int:
    """保存为 jsonl, 返回条数。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def _load_hf(repo: str, config: Optional[str] = None, split: str = "test", trust_remote_code: bool = False, **kw):
    """加载 HF dataset (带代理/镜像兼容)。trust_remote_code=True 用于老式 script 数据集。"""
    from datasets import load_dataset
    if HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    if config:
        return load_dataset(repo, config, split=split, trust_remote_code=trust_remote_code, **kw)
    return load_dataset(repo, split=split, trust_remote_code=trust_remote_code, **kw)


def _github_raw(url: str) -> List[dict]:
    """从 GitHub raw URL 下载 jsonl (每行一个 json)。"""
    text = _github_text(url)
    rows = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _github_text(url: str) -> str:
    """从 GitHub raw URL 下载文本 (csv/json 等任意文本)。"""
    proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers) if handlers else urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


# ====================================================================
# 各数据集下载 + 转换函数。每个返回 List[dict] (平台 jsonl 格式)。
# ====================================================================

def dl_mmlu() -> List[dict]:
    ds = _load_hf("cais/mmlu", "all", split="test")
    rows = []
    for i, r in enumerate(ds):
        rows.append({
            "sample_id": f"mmlu_{i}",
            "subject": r.get("subject", ""),
            "question": r["question"],
            "choices": r["choices"],
            "gold": "ABCD"[r["answer"]],
        })
    return rows


def dl_mmlu_pro() -> List[dict]:
    ds = _load_hf("TIGER-Lab/MMLU-Pro", split="test")
    rows = []
    for i, r in enumerate(ds):
        rows.append({
            "sample_id": f"mmlu_pro_{i}",
            "subject": r.get("category", ""),
            "question": r["question"],
            "choices": r["options"],
            "gold": "ABCDEFGHIJ"[r["answer_index"]],
        })
    return rows


def dl_gpqa() -> List[dict]:
    import random
    rng = random.Random(42)
    token = os.environ.get("HF_TOKEN")
    ds = _load_hf("Idavidrein/gpqa", "gpqa_diamond", split="train", token=token)
    rows = []
    for i, r in enumerate(ds):
        correct = r["Correct Answer"]
        wrongs = [r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        choices = [correct] + wrongs
        rng.shuffle(choices)
        gold = "ABCD"[choices.index(correct)]
        rows.append({
            "sample_id": f"gpqa_{i}",
            "subject": r.get("Subdomain", r.get("High-level domain", "")),
            "question": r["Question"],
            "choices": choices,
            "gold": gold,
        })
    return rows


def dl_ceval() -> List[dict]:
    from datasets import load_dataset
    # C-Eval 有 52 个 subject config, 全部加载
    subjects = [
        "computer_network", "operating_system", "computer_architecture", "college_programming",
        "college_physics", "college_chemistry", "advanced_mathematics", "probability_and_statistics",
        "discrete_mathematics", "electrical_engineer", "metrology_engineer", "high_school_mathematics",
        "high_school_physics", "high_school_chemistry", "high_school_biology", "middle_school_mathematics",
        "middle_school_biology", "middle_school_physics", "middle_school_chemistry", "veterinary_medicine",
        "college_economics", "business_administration", "marxism", "mao_zedong_thought",
        "education_science", "teacher_qualification", "high_school_politics", "high_school_geography",
        "middle_school_politics", "middle_school_geography", "modern_chinese_history", "ideological_and_moral_cultivation",
        "logic", "law", "chinese_language_and_literature", "art_studies",
        "professional_tour_guide", "legal_professional", "high_school_chinese", "high_school_history",
        "middle_school_history", "civil_servant", "sports_science", "plant_protection",
        "basic_medicine", "clinical_medicine", "urban_and_rural_planner", "accountant",
        "fire_engineer", "environmental_impact_assessment_engineer", "tax_accountant", "physician",
    ]
    rows = []
    idx = 0
    for subj in subjects:
        try:
            ds = load_dataset("ceval/ceval-exam", subj, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"    跳过 {subj}: {e}")
            continue
        for r in ds:
            choices = [r.get("A", ""), r.get("B", ""), r.get("C", ""), r.get("D", "")]
            ans = (r.get("answer") or "").strip()
            rows.append({
                "sample_id": f"ceval_{idx}",
                "subject": subj,
                "question": r.get("question", ""),
                "choices": choices,
                "gold": ans.upper() if ans else "",
            })
            idx += 1
    return rows


def dl_cmmlu() -> List[dict]:
    # haonan-li/cmmlu 的 HF script 版已不被新版 datasets 支持, 改用 GitHub csv 源。
    # csv 格式: 每行 [index, Question, A, B, C, D, Answer(字母)]
    import csv
    import io
    subjects = [
        "agronomy", "anatomy", "ancient_chinese", "arts", "astronomy", "business_ethics",
        "chinese_civilization", "chinese_driving_rule", "chinese_food_culture", "chinese_foreign_policy",
        "chinese_history", "chinese_literature", "chinese_teacher_qualification", "clinical_knowledge",
        "college_actuarial_science", "college_education", "college_engineering_hydrology", "college_law",
        "college_mathematics", "college_medical_statistics", "college_medicine", "computer_science",
        "computer_security", "conceptual_physics", "construction_project_management", "economics",
        "education", "electrical_engineering", "elementary_chinese", "elementary_commonsense",
        "elementary_information_and_technology", "elementary_mathematics", "ethnology", "food_science",
        "genetics", "global_facts", "high_school_biology", "high_school_chemistry",
        "high_school_geography", "high_school_mathematics", "high_school_physics", "high_school_politics",
        "human_body", "international_law", "journalism", "jurisprudence", "legal_and_moral_basis",
        "logical", "machine_learning", "management", "marketing", "marxist_theory", "modern_chinese",
        "nutrition", "philosophy", "professional_accounting", "professional_law", "professional_medicine",
        "professional_psychology", "public_relations", "security_study", "sociology", "sports_science",
        "traditional_chinese_medicine", "virology", "world_history", "world_religions",
    ]
    rows = []
    idx = 0
    for subj in subjects:
        url = f"https://raw.githubusercontent.com/haonan-li/CMMLU/master/data/test/{subj}.csv"
        try:
            text = _github_text(url)
        except Exception as e:
            print(f"    跳过 {subj}: {e}")
            continue
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)  # 表头: ,Question,A,B,C,D,Answer
        for r in reader:
            if len(r) < 7:
                continue
            # r = [index, Question, A, B, C, D, Answer]
            ans = (r[6] or "").strip()
            for ch in ans:
                if ch in "ABCD":
                    ans = ch; break
            rows.append({
                "sample_id": f"cmmlu_{idx}",
                "subject": subj,
                "question": r[1],
                "choices": [r[2], r[3], r[4], r[5]],
                "gold": ans.upper() if ans else "",
            })
            idx += 1
    return rows


def dl_arc() -> List[dict]:
    ds = _load_hf("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for i, r in enumerate(ds):
        ch = r["choices"]
        labels = ch["label"]
        texts = ch["text"]
        ans = r["answerKey"]
        # answerKey 通常是字母, 偶尔是数字 -> 映射到 label 下标
        gold = ""
        if ans in labels:
            gold = ans
        else:
            try:
                gold = labels[int(ans) - 1]
            except (ValueError, IndexError):
                gold = ""
        rows.append({
            "sample_id": f"arc_{i}",
            "question": r["question"],
            "choices": texts,
            "gold": gold,
        })
    return rows


def dl_hellaswag() -> List[dict]:
    ds = _load_hf("Rowan/hellaswag", split="validation")
    rows = []
    for i, r in enumerate(ds):
        ctx = r.get("ctx", "")
        # 拼活动描述 + 上下文, 让题面完整
        question = f"{r.get('activity_label', '')}: {ctx}"
        rows.append({
            "sample_id": f"hellaswag_{i}",
            "question": question,
            "choices": r["endings"],
            "gold": "ABCD"[int(r["label"])],
        })
    return rows


def dl_winogrande() -> List[dict]:
    ds = _load_hf("allenai/winogrande", "winogrande_xl", split="validation")
    rows = []
    for i, r in enumerate(ds):
        rows.append({
            "sample_id": f"winogrande_{i}",
            "question": r["sentence"],
            "choices": [r["option1"], r["option2"]],
            "gold": "AB"[int(r["answer"]) - 1],
        })
    return rows


def dl_truthfulqa() -> List[dict]:
    ds = _load_hf("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rows = []
    for i, r in enumerate(ds):
        mc1 = r["mc1_targets"]
        choices = mc1["choices"]
        labels = mc1["labels"]
        gold_idx = labels.index(1) if 1 in labels else 0
        rows.append({
            "sample_id": f"truthfulqa_{i}",
            "question": r["question"],
            "choices": choices,
            "gold": "ABCD"[gold_idx] if gold_idx < 4 else "A",
        })
    return rows


def dl_gsm8k() -> List[dict]:
    ds = _load_hf("openai/gsm8k", "main", split="test")
    rows = []
    for i, r in enumerate(ds):
        ans = r["answer"]
        # 提取 #### 后的数字
        gold = ""
        if "####" in ans:
            gold = ans.split("####")[-1].strip().replace(",", "")
        rows.append({
            "sample_id": f"gsm8k_{i}",
            "question": r["question"],
            "gold": gold,
        })
    return rows


def dl_math500() -> List[dict]:
    ds = _load_hf("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, r in enumerate(ds):
        rows.append({
            "sample_id": f"math500_{i}",
            "question": r["problem"],
            "gold": r["answer"],
            "subject": r.get("subject", ""),
        })
    return rows


def dl_aime() -> List[dict]:
    rows = []
    idx = 0
    # AIME 2024
    try:
        ds = _load_hf("Maxwell-Jia/AIME_2024", split="train")
        for r in ds:
            rows.append({
                "sample_id": f"aime_{idx}",
                "question": r["Problem"],
                "gold": str(r["Answer"]),
            })
            idx += 1
    except Exception as e:
        print(f"    AIME 2024 失败: {e}")
    # AIME 2025 (可选)
    try:
        ds = _load_hf("opencompass/AIME2025", "AIME2025-I", split="test")
        for r in ds:
            rows.append({
                "sample_id": f"aime_{idx}",
                "question": r.get("question", r.get("Problem", "")),
                "gold": str(r.get("answer", r.get("Answer", ""))),
            })
            idx += 1
    except Exception as e:
        print(f"    AIME 2025-I 跳过: {e}")
    try:
        ds = _load_hf("opencompass/AIME2025", "AIME2025-II", split="test")
        for r in ds:
            rows.append({
                "sample_id": f"aime_{idx}",
                "question": r.get("question", r.get("Problem", "")),
                "gold": str(r.get("answer", r.get("Answer", ""))),
            })
            idx += 1
    except Exception as e:
        print(f"    AIME 2025-II 跳过: {e}")
    return rows


def dl_humaneval() -> List[dict]:
    ds = _load_hf("openai/openai_humaneval", split="test")
    rows = []
    for i, r in enumerate(ds):
        rows.append({
            "sample_id": f"humaneval_{i}",
            "entry_point": r["entry_point"],
            "prompt": r["prompt"],
            "test_code": r["test"],
        })
    return rows


def dl_mbpp() -> List[dict]:
    ds = _load_hf("google-research-datasets/mbpp", "sanitized", split="test")
    rows = []
    for i, r in enumerate(ds):
        # sanitized 版 test_list 是 assert 字符串列表
        test_code = "\n".join(r.get("test_list", []))
        rows.append({
            "sample_id": f"mbpp_{i}",
            "entry_point": r.get("code", "").split("def ")[-1].split("(")[0] if r.get("code") else f"task_{r.get('task_id', i)}",
            "prompt": r.get("prompt", r.get("text", "")),
            "test_code": test_code,
        })
    return rows


def dl_ifeval() -> List[dict]:
    """IFEval: 把官方 instruction_id + kwargs 转成平台的 constraints 格式。"""
    ds = _load_hf("google/IFEval", split="train")
    rows = []
    for i, r in enumerate(ds):
        constraints = _ifeval_to_constraints(
            r.get("instruction_id_list", []),
            r.get("kwargs", []),
        )
        rows.append({
            "sample_id": f"ifeval_{i}",
            "question": r["prompt"],
            "constraints": constraints,
        })
    return rows


def _ifeval_to_constraints(inst_ids: List[str], kwargs_list: List[dict]) -> List[dict]:
    """把 IFEval 官方 instruction_id + kwargs 映射成平台 rule_check 支持的约束类型。

    IFEval 有 25 类指令, 每类有特定 kwargs。这里基于 instruction_id 精确映射,
    并把 kwargs 原样带入 args (check_instruction 会按类型读取对应字段)。
    """
    # instruction_id -> 平台约束 type 的映射表
    _MAP = {
        "punctuation:no_comma": "no_commas",
        "detectable_format:number_highlighted_sections": "num_highlights",
        "detectable_format:number_bullet_lists": "num_bullets",
        "detectable_format:title": "markdown_title",
        "detectable_format:multiple_sections": "multiple_sections",
        "detectable_format:json_format": "is_json",
        "detectable_format:constrained_response": "constrained_response",
        "detectable_content:number_placeholders": "num_placeholders",
        "detectable_content:postscript": "postscript",
        "keywords:existence": "contains_keyword",
        "keywords:forbidden_words": "forbidden_words",
        "keywords:frequency": "keyword_frequency",
        "keywords:letter_frequency": "letter_frequency",
        "length_constraints:number_words": None,        # 按关系选 max/min/exact
        "length_constraints:number_sentences": None,    # 按关系选 max/min
        "length_constraints:number_paragraphs": "min_paragraphs",
        "length_constraints:nth_paragraph_first_word": "nth_paragraph_word",
        "change_case:english_capital": "all_uppercase",
        "change_case:english_lowercase": "all_lowercase",
        "change_case:capital_word_frequency": "capital_frequency",
        "startend:end_checker": "endswith",
        "startend:quotation": "quotation",
        "combination:two_responses": "two_responses",
        "combination:repeat_prompt": "repeat_prompt",
        "language:response_language": "response_language",
    }
    cons = []
    for inst_id, kwargs in zip(inst_ids, kwargs_list):
        kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
        ctype = _MAP.get(inst_id)
        if inst_id == "length_constraints:number_words":
            rel = kwargs.get("relation", "at least")
            if rel == "less than":
                ctype = "max_words"
            elif rel == "equal to":
                ctype = "exact_words"
            else:
                ctype = "min_words"
        elif inst_id == "length_constraints:number_sentences":
            rel = kwargs.get("relation", "less than")
            ctype = "max_sentences" if rel == "less than" else "min_sentences"
        if ctype:
            cons.append({"type": ctype, "args": kwargs, "instruction_id": inst_id})
    return cons


def dl_mt_bench() -> List[dict]:
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    raw = _github_raw(url)
    rows = []
    for r in raw:
        turns = r.get("turns", [])
        question = "\n\n".join(turns) if turns else ""
        ref = r.get("reference", [])
        rows.append({
            "sample_id": f"mt_bench_{r.get('question_id', len(rows))}",
            "question": question,
            "reference": ref if isinstance(ref, list) else [str(ref)],
            "turn": len(turns),
            "category": r.get("category", ""),
        })
    return rows


def dl_alpaca_eval() -> List[dict]:
    # tatsu-lab/alpaca_eval 的 HF script 版已不支持, 直接下 alpaca_eval.json
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    import json as _json
    data = _json.load(open(p, encoding="utf-8"))
    rows = []
    for i, r in enumerate(data):
        rows.append({
            "sample_id": f"alpaca_eval_{i}",
            "question": r.get("instruction", ""),
            "reference": [r.get("output", "")],
            "generator": r.get("generator", ""),
        })
    return rows


def dl_arena_hard() -> List[dict]:
    # v2.0 (750 条, 新版); 路径是 data/arena-hard-v2.0/question.jsonl (注意无重复子目录)
    url = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v2.0/question.jsonl"
    raw = _github_raw(url)
    rows = []
    for r in raw:
        rows.append({
            "sample_id": f"arena_hard_{str(r.get('uid', len(rows)))[:8]}",
            "question": r.get("prompt", ""),
            "cluster": r.get("cluster", ""),
            "category": r.get("category", ""),
        })
    return rows


# ====================================================================
# 新增 benchmark (2025-2026 新模型官方评测用的)
# ====================================================================

def dl_supergpqa() -> List[dict]:
    """SuperGPQA: 研究生级多学科选择题 (m-a-p/SuperGPQA)"""
    ds = _load_hf("m-a-p/SuperGPQA", split="train")
    rows = []
    for i, r in enumerate(ds):
        # options 是 list, answer_letter 是 A/B/...
        choices = r.get("options", [])
        gold = (r.get("answer_letter") or "").strip().upper()
        rows.append({
            "sample_id": f"supergpqa_{i}",
            "subject": r.get("field", r.get("discipline", "")),
            "question": r.get("question", ""),
            "choices": choices,
            "gold": gold,
        })
    return rows


def dl_mmlu_redux() -> List[dict]:
    """MMLU-Redux: MMLU 去噪版 (edinburgh-dawg/mmlu-redux-2.0), 多学科 config"""
    from datasets import get_dataset_config_names, load_dataset
    cfgs = get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0")
    rows = []
    idx = 0
    for cfg in cfgs:
        try:
            ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", cfg, split="test")
        except Exception as e:
            print(f"    跳过 {cfg}: {e}")
            continue
        for r in ds:
            choices = r.get("choices", [])
            # answer 可能是 int (下标) 或字母; correct_answer 也有
            ans = r.get("answer")
            if isinstance(ans, int):
                gold = "ABCD"[ans] if 0 <= ans < len(choices) else ""
            else:
                gold = str(ans or r.get("correct_answer") or "").strip().upper()
            rows.append({
                "sample_id": f"mmlu_redux_{idx}",
                "subject": cfg,
                "question": r.get("question", ""),
                "choices": choices,
                "gold": gold,
            })
            idx += 1
    return rows


def dl_simpleqa() -> List[dict]:
    """SimpleQA: 简短问答 (google/simpleqa-verified), 精确匹配"""
    ds = _load_hf("google/simpleqa-verified", split="eval")
    rows = []
    for i, r in enumerate(ds):
        rows.append({
            "sample_id": f"simpleqa_{i}",
            "question": r.get("problem", ""),
            "gold": r.get("answer", ""),
            "topic": r.get("topic", ""),
        })
    return rows


def dl_bbh() -> List[dict]:
    """BBH (BigBenchHard): 推理 (lukaemon/bbh, 每子任务一个 parquet)"""
    from huggingface_hub import HfApi
    from datasets import load_dataset
    api = HfApi()
    files = api.list_repo_files("lukaemon/bbh", repo_type="dataset")
    tasks = sorted({f.split("/")[0] for f in files if f.endswith(".parquet")})
    rows = []
    idx = 0
    for task in tasks:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
        except Exception as e:
            print(f"    跳过 {task}: {e}")
            continue
        for r in ds:
            # 字段: input (题面), target (答案)
            rows.append({
                "sample_id": f"bbh_{idx}",
                "subject": task,
                "question": r.get("input", ""),
                "gold": r.get("target", ""),
            })
            idx += 1
    return rows


def dl_agieval() -> List[dict]:
    """AGIEval: 综合能力 (用 dmayhem93/agieval-* parquet 镜像, 按科目)"""
    from huggingface_hub import HfApi
    api = HfApi()
    res = list(api.list_datasets(search="agieval", limit=50))
    subjects = [d.id for d in res if d.id.startswith("dmayhem93/agieval-")]
    rows = []
    idx = 0
    for subj_repo in subjects:
        try:
            from datasets import load_dataset
            ds = load_dataset(subj_repo, split="test")
        except Exception as e:
            print(f"    跳过 {subj_repo}: {e}")
            continue
        for r in ds:
            # AGIEval 字段: query (含题面), choices, gold (可能是 list[int] 下标 或 字母)
            q = r.get("query", r.get("question", ""))
            choices = r.get("choices", [])
            raw_gold = r.get("gold")
            gold = ""
            if isinstance(raw_gold, list) and raw_gold:
                # gold 是下标列表, 取第一个
                gi = raw_gold[0]
                gold = "ABCDEFGHIJ"[gi] if isinstance(gi, int) and 0 <= gi < len(choices) else str(gi)
            elif isinstance(raw_gold, int):
                gold = "ABCDEFGHIJ"[raw_gold] if 0 <= raw_gold < len(choices) else ""
            elif raw_gold:
                gold = str(raw_gold).strip().upper()
                if gold.isdigit():
                    gold = "ABCDEFGHIJ"[int(gold)]
            rows.append({
                "sample_id": f"agieval_{idx}",
                "subject": subj_repo.replace("dmayhem93/agieval-", ""),
                "question": q,
                "choices": choices,
                "gold": gold,
            })
            idx += 1
    return rows


def dl_livecodebench() -> List[dict]:
    """LiveCodeBench: 代码生成 (livecodebench/test_generation parquet)

    字段: question_content, starter_code, function_name, test (JSON串:
    [{"input": "<参数JSON列表>", "output": "<期望返回>", "testtype": "functional"}])
    test 按原样保存 (插件运行时解析为函数调用断言)。
    """
    from datasets import load_dataset
    ds = load_dataset("livecodebench/test_generation", split="test", data_files={"test": "test.parquet"})
    rows = []
    for i, r in enumerate(ds):
        rows.append({
            "sample_id": f"livecodebench_{i}",
            "entry_point": r.get("function_name", f"lcb_{i}"),
            "prompt": r.get("question_content", ""),
            "test_code": r.get("test", ""),  # JSON 串, 由插件解析
        })
    return rows



def dl_bigcodebench() -> List[dict]:
    """BigCodeBench: 代码生成 (bigcode/bigcodebench parquet, split=v0.1.4)"""
    from datasets import load_dataset
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    rows = []
    for i, r in enumerate(ds):
        # 字段: task_id, instruct_prompt, code_prompt, test, entry_point
        rows.append({
            "sample_id": f"bigcodebench_{i}",
            "entry_point": r.get("entry_point", f"bcb_{i}"),
            "prompt": r.get("instruct_prompt", r.get("code_prompt", "")),
            "test_code": r.get("test", ""),
        })
    return rows


def dl_hle() -> List[dict]:
    """HLE (Humanity's Last Exam): 超难知识题 (cais/hle, gated 需 HF_TOKEN)"""
    token = os.environ.get("HF_TOKEN")
    ds = _load_hf("cais/hle", split="test", token=token)
    rows = []
    for i, r in enumerate(ds):
        # 字段: question, choices, answer, image (文本题无图)
        choices = r.get("choices", [])
        gold = (r.get("answer") or "").strip()
        # answer 可能是字母或文本
        if gold and gold in "ABCDEFGHIJ":
            pass
        rows.append({
            "sample_id": f"hle_{i}",
            "question": r.get("question", ""),
            "choices": choices,
            "gold": gold.upper() if gold and len(gold) == 1 else gold,
            "subject": r.get("field", r.get("category", "")),
        })
    return rows


def dl_evalplus() -> List[dict]:
    """EvalPlus HumanEval+: HumanEval 增强版, 测试用例更多更严 (evalplus/humanevalplus)"""
    ds = _load_hf("evalplus/humanevalplus", split="test")
    rows = []
    for r in ds:
        rows.append({
            "sample_id": r.get("task_id", ""),
            "prompt": r.get("prompt", ""),
            "canonical_solution": r.get("canonical_solution", ""),
            "entry_point": r.get("entry_point", ""),
            "test_code": r.get("test", ""),
        })
    return rows


def dl_drop() -> List[dict]:
    """DROP: 阅读理解+离散推理 (drop, validation split)"""
    ds = _load_hf("drop", split="validation", trust_remote_code=True)
    rows = []
    for i, r in enumerate(ds):
        ans = r.get("answers_spans", {})
        spans = ans.get("spans", []) if isinstance(ans, dict) else []
        rows.append({
            "sample_id": f"drop_{i}",
            "passage": r.get("passage", ""),
            "question": r.get("question", ""),
            "gold": spans,  # 答案列表, 任一匹配即对
        })
    return rows


def dl_ds1000() -> List[dict]:
    """DS-1000: 数据科学代码生成 (xlangai/DS-1000, pandas/numpy/matplotlib/scipy)"""
    ds = _load_hf("xlangai/DS-1000", split="test")
    rows = []
    for r in ds:
        meta = r.get("metadata", {}) or {}
        rows.append({
            "sample_id": f"ds1000_{meta.get('problem_id', '')}",
            "prompt": r.get("prompt", ""),
            "reference_code": r.get("reference_code", ""),
            "code_context": r.get("code_context", ""),
            "library": meta.get("library", ""),
        })
    return rows


def dl_longbench_v2() -> List[dict]:
    """LongBench-V2: 长文本理解 (THUDM/LongBench-v2, 4选项长上下文MCQ)"""
    ds = _load_hf("THUDM/LongBench-v2", split="train")
    rows = []
    for r in ds:
        rows.append({
            "sample_id": r.get("_id", ""),
            "context": r.get("context", ""),
            "question": r.get("question", ""),
            "choices": [r.get("choice_A", ""), r.get("choice_B", ""),
                        r.get("choice_C", ""), r.get("choice_D", "")],
            "gold": (r.get("answer") or "").strip().upper(),
            "subject": r.get("domain", ""),
            "difficulty": r.get("difficulty", ""),
            "length": r.get("length", ""),
        })
    return rows


def dl_mrcr() -> List[dict]:
    """MRCR: 多轮长上下文检索 (openai/mrcr, needle-in-a-haystack)"""
    from datasets import get_dataset_split_names, load_dataset
    splits = get_dataset_split_names("openai/mrcr")
    split = "test" if "test" in splits else splits[0]
    ds = load_dataset("openai/mrcr", split=split)
    rows = []
    for i, r in enumerate(ds):
        # prompt 是 JSON 消息列表字符串, 拼成纯文本上下文
        prompt_str = r.get("prompt", "")
        context = _mrcr_msgs_to_text(prompt_str)
        rows.append({
            "sample_id": f"mrcr_{i}",
            "context": context,
            "gold": r.get("answer", ""),
        })
        if i >= 2000:  # 限制条数, 避免过大
            break
    return rows


def _mrcr_msgs_to_text(prompt_str: str) -> str:
    """把 MRCR 的 JSON 消息列表拼成纯文本 (role: content 换行)。"""
    if not prompt_str:
        return ""
    try:
        msgs = json.loads(prompt_str)
    except (json.JSONDecodeError, TypeError):
        return prompt_str
    parts = []
    for m in msgs:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        else:
            parts.append(str(m))
    return "\n\n".join(parts)


def dl_corpusqa() -> List[dict]:
    """CorpusQA: 超长语料问答 (Tongyi-Zhiwen/CorpusQA, 1m_4domains.jsonl)

    HF 原始结构 (无 context 字段, 语料在 prompt 消息列表里):
      - prompt: [{role:'system', content:'You are...'},
                 {role:'user',   content:'# Document 1:\\n...'}]  <- user.content 是超长语料
      - question: str
      - answer: list[str]   多个正确答案 (任一匹配即对)
      - set/domain/doc_files 等元信息
    之前误用 r.get('context') 取不到任何字段 -> context 全空, 模型收不到语料。
    """
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    p = hf_hub_download("Tongyi-Zhiwen/CorpusQA", "1m_4domains.jsonl",
                        repo_type="dataset", token=token)
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # 语料在 prompt 的 user 消息 content 里 (system 是固定提示, 不含语料)。
            context = ""
            for m in r.get("prompt", []):
                if m.get("role") == "user":
                    context = m.get("content", "")
                    break
            # answer 是 list (多个正确答案), 保持 list 结构; evaluate 侧按"任一匹配"判分。
            gold = r.get("answer", r.get("gold", ""))
            if isinstance(gold, str):
                gold = [gold] if gold else []
            rows.append({
                "sample_id": r.get("id") or f"corpusqa_{i}",
                "context": context,
                "question": r.get("question", r.get("query", "")),
                "gold": gold,
            })
            if i >= 2000:
                break
    return rows


def dl_bfcl() -> List[dict]:
    """BFCL: 函数调用评测 (gorilla-llm/Berkeley-Function-Calling-Leaderboard, simple 子集)"""
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    # 测试集 + 对应 possible_answer
    q_path = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                             "BFCL_v3_simple.json", repo_type="dataset", token=token)
    a_path = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                             "possible_answer/BFCL_v3_simple.json", repo_type="dataset", token=token)
    questions = [json.loads(l) for l in open(q_path) if l.strip()]
    answers = {json.loads(l)["id"]: json.loads(l)["ground_truth"]
               for l in open(a_path) if l.strip()}
    rows = []
    for r in questions:
        q = r.get("question", [])
        # question 是 [[{role,content}]], 取用户消息文本
        user_msg = ""
        if q and isinstance(q, list) and isinstance(q[0], list) and q[0]:
            user_msg = q[0][0].get("content", "") if isinstance(q[0][0], dict) else str(q[0][0])
        rows.append({
            "sample_id": r.get("id", ""),
            "question": user_msg,
            "functions": r.get("function", []),
            "gold": answers.get(r.get("id"), []),
        })
    return rows


def dl_swebench() -> List[dict]:
    """SWE-bench Verified: 代码 Agent 修复真实 GitHub issue (princeton-nlp/SWE-bench_Verified)"""
    token = os.environ.get("HF_TOKEN")
    ds = _load_hf("princeton-nlp/SWE-bench_Verified", split="test", token=token)
    rows = []
    for r in ds:
        rows.append({
            "sample_id": r.get("instance_id", ""),
            "repo": r.get("repo", ""),
            "base_commit": r.get("base_commit", ""),
            "problem_statement": r.get("problem_statement", ""),
            "hints_text": r.get("hints_text", ""),
            "patch": r.get("patch", ""),
            "test_patch": r.get("test_patch", ""),
            "fail_to_pass": r.get("FAIL_TO_PASS", ""),
            "pass_to_pass": r.get("PASS_TO_PASS", ""),
            "version": r.get("version", ""),
            "environment_setup_commit": r.get("environment_setup_commit", ""),
            "difficulty": r.get("difficulty", ""),
        })
    return rows


# 数据集注册表: name -> (下载函数, 预计条数)
DATASETS = {
    "mmlu":        (dl_mmlu,        14042),
    "mmlu_pro":    (dl_mmlu_pro,    12032),
    "gpqa":        (dl_gpqa,        198),
    "ceval":       (dl_ceval,       13942),
    "cmmlu":       (dl_cmmlu,       11528),
    "arc":         (dl_arc,         1172),
    "hellaswag":   (dl_hellaswag,   10042),
    "winogrande":  (dl_winogrande,  1267),
    "truthfulqa":  (dl_truthfulqa,  817),
    "gsm8k":       (dl_gsm8k,       1319),
    "math500":     (dl_math500,     500),
    "aime":        (dl_aime,        30),
    "humaneval":   (dl_humaneval,   164),
    "mbpp":        (dl_mbpp,        427),
    "ifeval":      (dl_ifeval,      541),
    "mt_bench":    (dl_mt_bench,    80),
    "alpaca_eval": (dl_alpaca_eval, 805),
    "arena_hard":  (dl_arena_hard,  750),
    # 新增 (2025-2026 新模型官方评测用)
    "supergpqa":     (dl_supergpqa,     26529),
    "mmlu_redux":    (dl_mmlu_redux,    14042),
    "simpleqa":      (dl_simpleqa,      4326),
    "bbh":           (dl_bbh,           6511),
    "agieval":       (dl_agieval,       17125),
    "livecodebench": (dl_livecodebench, 140),
    "bigcodebench":  (dl_bigcodebench,  1140),
    "hle":           (dl_hle,           3000),
    # 第二批新增 (前沿模型 agent/代码/长文本评测)
    "evalplus":      (dl_evalplus,      164),
    "drop":          (dl_drop,          9535),
    "ds1000":        (dl_ds1000,        1000),
    "longbench_v2":  (dl_longbench_v2,  503),
    "mrcr":          (dl_mrcr,          2000),
    "corpusqa":      (dl_corpusqa,      2000),
    "bfcl":          (dl_bfcl,          400),
    "swebench":      (dl_swebench,      500),
}


def main():
    if "--list" in sys.argv:
        print(f"共 {len(DATASETS)} 个数据集:")
        for name, (fn, n) in DATASETS.items():
            print(f"  {name:<14} ~{n} 条")
        return

    targets = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not targets:
        targets = list(DATASETS.keys())

    print(f"将下载 {len(targets)} 个数据集到 {DATA_DIR}/\n")
    success, failed = [], []
    for name in targets:
        if name not in DATASETS:
            print(f"  ✗ 未知数据集: {name} (用 --list 查看)"); failed.append(name); continue
        fn, expect = DATASETS[name]
        print(f"▶ {name} (预计 ~{expect} 条)...")
        try:
            rows = fn()
            n = _save_jsonl(name, rows)
            print(f"  ✓ {name}: {n} 条 已保存\n")
            success.append(name)
        except Exception as e:
            print(f"  ✗ {name} 失败: {e}\n")
            failed.append(name)

    print(f"\n完成: 成功 {len(success)} ({', '.join(success)}), 失败 {len(failed)} ({', '.join(failed) or '无'})")


if __name__ == "__main__":
    main()
