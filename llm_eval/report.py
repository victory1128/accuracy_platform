"""报告生成

输出:
- JSON: 完整结果 (含每条样本)
- HTML: 可视化仪表板 (分数总览 + 乱码分析 + 样例)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from html import escape
from typing import Any, Dict, List, Optional

from .models import RunResult, Stage, TaskType

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>精度测试报告 - {model_name}</title>
<style>
:root {{
  --bg: #f8fafc; --card: #ffffff; --border: #e2e8f0;
  --fg: #0f172a; --muted: #64748b; --accent: #4f46e5; --good: #16a34a; --warn: #d97706; --bad: #dc2626;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--fg); line-height: 1.6; }}
.container {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 18px; margin: 28px 0 12px; border-left: 3px solid var(--accent); padding-left: 10px; }}
.meta {{ color: var(--muted); font-size: 13px; margin-bottom: 20px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 10px; margin-bottom: 8px; }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }}
.card .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }}
.card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
.card .sub {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 10px; overflow: hidden; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }}
th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }}
tr:hover {{ background: rgba(79,158,255,.06); }}
.score {{ font-weight: 700; }}
.tag {{ display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 10px; background: rgba(79,158,255,.15); color: var(--accent); margin-right: 4px; }}
.tag.pretrain {{ background: rgba(74,222,128,.15); color: var(--good); }}
.tag.posttrain {{ background: rgba(251,191,36,.15); color: var(--warn); }}
.bar {{ height: 6px; border-radius: 3px; background: var(--border); overflow: hidden; margin-top: 4px; }}
.bar > div {{ height: 100%; background: var(--accent); }}
.badge {{ font-size: 11px; padding: 2px 7px; border-radius: 6px; }}
.badge.good {{ background: rgba(74,222,128,.15); color: var(--good); }}
.badge.warn {{ background: rgba(251,191,36,.15); color: var(--warn); }}
.badge.bad {{ background: rgba(248,113,113,.15); color: var(--bad); }}
.sample {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 10px; }}
.sample pre {{ white-space: pre-wrap; word-break: break-word; background: #f1f5f9; padding: 10px; border-radius: 6px; font-size: 12px; max-height: 200px; overflow: auto; margin: 6px 0; }}
.diag {{ font-size: 12px; color: var(--warn); }}
.footer {{ color: var(--muted); font-size: 12px; margin-top: 30px; text-align: center; }}
.hint {{ color: var(--muted); font-size: 12px; margin-bottom: 12px; }}
/* 交互式请求明细表格 (复刻请求浏览器) */
.req-toolbar {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px; margin-bottom: 12px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
.req-toolbar input, .req-toolbar select {{ background: #f1f5f9; border: 1px solid var(--border); color: var(--fg); border-radius: 6px; padding: 8px 10px; font-size: 13px; }}
.req-toolbar input[type=text] {{ flex: 1; min-width: 240px; }}
.req-toolbar label {{ font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 4px; }}
/* 分页控件 */
.rpt-pager {{ display: flex; gap: 6px; align-items: center; justify-content: center; margin: 12px 0; flex-wrap: wrap; }}
.rpt-pager button {{ background: var(--card); border: 1px solid var(--border); color: var(--fg); border-radius: 6px; padding: 6px 12px; font-size: 12px; cursor: pointer; }}
.rpt-pager button:hover:not(:disabled) {{ border-color: var(--accent); color: var(--accent); }}
.rpt-pager button:disabled {{ opacity: .4; cursor: default; }}
.rpt-pager button.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.rpt-pager .rpt-jump {{ display: flex; align-items: center; gap: 4px; font-size: 12px; color: var(--muted); }}
.rpt-pager input[type=number] {{ width: 56px; background: #f1f5f9; border: 1px solid var(--border); color: var(--fg); border-radius: 6px; padding: 6px 8px; font-size: 12px; text-align: center; }}
.rpt-table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 10px; overflow: hidden; }}
.rpt-table th, .rpt-table td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); font-size: 12px; vertical-align: top; }}
.rpt-table th {{ color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; position: sticky; top: 0; background: #f1f5f9; }}
.rpt-table tr:hover {{ background: rgba(79,70,229,.06); }}
.rpt-hash {{ font-family: ui-monospace,monospace; color: var(--accent); cursor: pointer; font-size: 11px; }}
.rpt-hash:hover {{ text-decoration: underline; }}
.rpt-detail {{ background: #f8fafc; }}
.rpt-detail-inner {{ padding: 14px; }}
.rpt-detail h5 {{ margin: 12px 0 4px; font-size: 12px; color: var(--accent); }}
.rpt-detail h5.reason {{ color: #7c3aed; }}
.rpt-detail h5.out {{ color: var(--good); }}
.rpt-detail h5:first-child {{ margin-top: 0; }}
.rpt-detail pre {{ white-space: pre-wrap; word-break: break-word; background: #ffffff; padding: 8px; border-radius: 6px; font-size: 11px; max-height: 240px; overflow: auto; margin: 4px 0; border: 1px solid var(--border); }}
.rpt-kv {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px; color: var(--muted); margin-bottom: 6px; }}
.rpt-kv b {{ color: var(--fg); font-weight: 600; }}
.rpt-mono {{ font-family: ui-monospace,monospace; }}
.empty {{ text-align: center; color: var(--muted); padding: 30px; }}
.warn-banner {{ background: rgba(217,119,6,.1); border: 1px solid var(--warn); color: var(--warn); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; }}
.warn-banner b {{ font-weight: 700; }}
.btn-back {{ background: var(--card); border: 1px solid var(--border); color: var(--accent); border-radius: 8px; padding: 8px 16px; font-size: 13px; cursor: pointer; margin-bottom: 16px; }}
.btn-back:hover {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.bench-link {{ color: var(--accent); cursor: pointer; text-decoration: none; }}
.bench-link:hover {{ text-decoration: underline; }}
/* IFEval 逐条约束明细小表 */
.constraint-table {{ width: 100%; border-collapse: collapse; margin: 4px 0 8px; font-size: 11px; }}
.constraint-table th, .constraint-table td {{ padding: 4px 8px; border-bottom: 1px solid var(--border); text-align: left; }}
.constraint-table th {{ color: var(--muted); font-weight: 600; background: #f1f5f9; }}
.constraint-table .ok {{ color: var(--good); font-weight: 700; }}
.constraint-table .no {{ color: var(--bad); font-weight: 700; }}
</style>
</head>
<body>
<div class="container">
  <h1>精度测试报告</h1>
  <div class="meta">模型: <b>{model_name}</b> · {benchmarks_count} 个评测集 · {started} → {finished}</div>
  {fake_stream_banner}

  <!-- ============ 主视图 ============ -->
  <div id="view-overview">
    <h2>总览</h2>
    <div class="cards">
      {summary_cards}
    </div>

    <h2>评测集明细 (分数 / 乱码 / 速度 / 延迟 / token / 长度{perf_extra_title})</h2>
    <div class="hint">点击评测集名称查看该集每条请求明细。</div>
    <table>
      <thead><tr>{bench_header}</tr></thead>
      <tbody>{bench_rows}</tbody>
    </table>

    <h2>乱码 / 异常输出分析</h2>
    <div class="cards">{gibberish_cards}</div>

    <h2>可疑样本示例 (最多5条)</h2>
    {suspicious_samples}
  </div>

  <!-- ============ 单评测集明细视图 ============ -->
  <div id="view-bench" style="display:none">
    <button class="btn-back" onclick="showOverview()">← 返回主报告</button>
    <div id="bench-summary"></div>
    <h2 id="bench-title">每条请求明细</h2>
    <div class="hint" id="bench-hint"></div>
    {request_details}
  </div>

  <div class="footer">由 精度测试平台 生成 · {generated_at}</div>
</div>
<script>
// 交互式请求明细渲染 (单页多视图: 主视图 + 各评测集明细视图)
// 明细视图按 task_type 用不同列模板, 每类只放该类关心的关键信息:
//   mcq  : 题面 / 标准答案 / 模型选择 / 对错
//   gen  : 题面 / 标准答案 / 抽取答案 / 对错
//   code : 题面 / 执行结果(pass/fail) / 错误信息
//   judge: 题面 / 参考答案 / 裁判分
//   rule : 题面 / 满足率 / 约束(满足/总数) / 逐条约束明细(展开)
// 性能列(速度/延迟/token/字符) 通用追加。
// 请求明细表格渲染逻辑抽到 rpt_table.js (与 SPA 任务详情"查看明细"页共用同一套)。
// 报告为自包含 HTML, 故把 rpt_table.js 内容内联进来 (占位符在 _render_html 末尾替换, 此处即其内容)。
__RPT_TABLE_JS__
(function() {{
  const DATA = window.RPT_REQS || [];
  const BENCHES = window.RPT_BENCHES || [];
  const BENCH_INFOS = window.RPT_BENCH_INFOS || {{}};
  // 表格挂到 #rpt-host (request_details 里包了工具栏+表格+分页的容器),
  // initRptTable 会重建其 innerHTML, 不影响 #bench-summary/#bench-title 等。
  const host = document.getElementById('rpt-host');
  const tableCtrl = initRptTable(host, DATA, BENCHES, BENCH_INFOS, {{ showToolbar: true }});

  // Ctrl/Cmd+A 范围收敛: 在明细 <pre> 或可编辑元素里按"全选"时只选该元素内容,
  // 而非整个页面 (复制输入/输出/思维链时, 默认 Ctrl+A 会选中文档全部, 很难用)。
  // pre 默认不可聚焦, 故用当前选区锚点定位所在 <pre>。
  window.addEventListener('keydown', function(e) {{
    if (!((e.ctrlKey || e.metaKey) && (e.key === 'a' || e.key === 'A'))) return;
    const ae = document.activeElement;
    if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA' || ae.isContentEditable)) {{
      const s = window.getSelection(); const r = document.createRange();
      r.selectNodeContents(ae); s.removeAllRanges(); s.addRange(r); e.preventDefault(); return;
    }}
    const s = window.getSelection && window.getSelection();
    if (s && s.rangeCount > 0) {{
      let n = s.anchorNode;
      while (n && n.tagName !== 'PRE') n = n.parentNode;
      if (n && n.tagName === 'PRE') {{
        const r = document.createRange(); r.selectNodeContents(n);
        s.removeAllRanges(); s.addRange(r); e.preventDefault();
      }}
    }}
  }});

  // ---- 视图切换 (报告特有: 主视图/单集明细视图) ----
  function card(label, value, sub) {{
    return '<div class="card"><div class="label">'+esc(label)+'</div><div class="value">'+esc(value)+'</div><div class="sub">'+esc(sub)+'</div></div>';
  }}
  window.showBench = function(name) {{
    document.getElementById('view-overview').style.display = 'none';
    document.getElementById('view-bench').style.display = '';
    const info = BENCH_INFOS[name] || {{}};
    const g = info.gibberish || {{}};
    document.getElementById('bench-title').textContent = (info.display||name) + ' · 每条请求明细';
    const hint = document.getElementById('bench-hint');
    hint.textContent = '类型 '+(info.task_type||'-')+' · 阶段 '+(info.stage||'-')+' · 样本 '+(info.num_samples||0)
      +' · 分数 '+(info.score_label||'-')+' · 乱码率 '+((g.suspicious_rate||0)*100).toFixed(1)+'% ('+g.overall_grade+')'
      + ' · 点击 Hash 展开看完整输入/思维链/输出';
    document.getElementById('bench-summary').innerHTML = '<div class="cards">'
      + card('分数', info.score_label||'-', (info.task_type||'')+' · '+(info.stage||''))
      + card('样本数', String(info.num_samples||0), '每条请求明细')
      + card('乱码率', ((g.suspicious_rate||0)*100).toFixed(1)+'%', (g.suspicious_count||0)+' 条可疑')
      + card('输出异常', String(g.abnormal_count||0), '截断/空输出')
      + '</div>';
    // 切到该集 (initRptTable 会设下拉值 + 重置页码 + 重渲染)
    tableCtrl.goto(name);
    window.scrollTo({{top:0, behavior:'smooth'}});
  }};
  window.showOverview = function() {{
    document.getElementById('view-bench').style.display = 'none';
    document.getElementById('view-overview').style.display = '';
    window.scrollTo({{top:0, behavior:'smooth'}});
  }};
}})();
</script>
</body>
</html>
"""


def save_json(run_results: List[RunResult], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "model": run_results[0].model_name if run_results else "",
        "generated_at": datetime.now().isoformat(),
        "results": [r.to_dict() for r in run_results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def save_html(run_results: List[RunResult], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    html = _render_html(run_results)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _score_value(r: RunResult) -> Optional[float]:
    """统一取出一个 0~100 的分数用于排序/展示"""
    agg = r.aggregate
    if "score_100" in agg and agg["score_100"] is not None:
        return agg["score_100"]
    if "mean_score" in agg and agg["mean_score"] is not None:
        return round(agg["mean_score"] * 10, 2)
    for k in ("accuracy", "pass_at_1", "instruction_following_rate"):
        if k in agg and agg[k] is not None:
            return round(agg[k] * 100, 2)
    return None


def _score_label(r: RunResult) -> str:
    v = _score_value(r)
    if v is None:
        return "-"
    # 统一纯数值 (0-100 量纲, 无百分号/分母), 与界面 summary 口径一致。
    return f"{v:g}"


def build_request_data(run_results: List[RunResult], lite: bool = False):
    """把 RunResult 列表摊平成请求明细数据 (报告 HTML 与 SPA 任务详情"查看明细"页共用)。

    返回 (flat_reqs, bench_infos, bench_options):
    - flat_reqs: 每条请求一个 dict (含题面/prompt/响应/思维链/答案/性能/乱码等), 供 rpt_table.js 渲染。
    - bench_infos: {benchmark: {display, task_type, stage, num_samples, score_label, gibberish, ...}}
    - bench_options: [{internal, display}] 供评测集下拉。

    lite=True (SPA 任务详情"查看明细"页用): 大文本字段 (prompt/response/reasoning_content/
    system_prompt) 不返回, question/reference/extracted/error 截断到预览长度, ifeval 只留汇总不含
    逐条 details。表格行只需这些摘要字段; 点 Hash 展开时由 /requests/{sample_id} 按需拉完整文本。
    报告 HTML 预嵌全部数据, 用 lite=False (默认)。
    """
    flat_reqs = []
    for r in run_results:
        meta = r.benchmark_meta
        for s in r.results:
            analysis = s.analysis or {}
            usage = s.usage or {}
            question = s.question or ""
            extracted = s.extracted or ""
            reference = s.reference or ""
            error = s.error or ""
            reasoning = s.reasoning_content or ""
            ifeval = analysis.get("ifeval") or {}
            row = {
                "request_hash": s.request_hash or "",
                "benchmark": r.benchmark,
                "benchmark_display": meta.display_name,
                "stage": meta.stage.value,
                "task_type": meta.task_type.value,
                "sample_id": s.sample_id or "",
                "question": question[:120] if lite else question,
                "prompt": (s.prompt or s.question or "") if not lite else "",
                "system_prompt": (s.system_prompt or "") if not lite else "",
                "response": (s.response or "") if not lite else "",
                "reasoning_content": reasoning if not lite else "",
                "has_reasoning": bool(reasoning),
                "reasoning_chars": len(reasoning),
                "extracted": extracted[:200] if lite else extracted,
                "gold": s.gold or "",
                "reference": reference[:200] if lite else reference,
                "correct": s.correct,
                "score": s.score,
                "error": error[:200] if lite else error,
                "latency_ms": s.latency_ms,
                "ttft_ms": s.ttft_ms,
                "tpot_ms": s.tpot_ms,
                "gen_time_ms": s.gen_time_ms,
                "streaming": s.streaming,
                "real_stream": s.real_stream,
                "tokens_per_sec": s.tokens_per_sec,
                "prompt_tokens": s.prompt_tokens,
                "completion_tokens": s.completion_tokens,
                "reasoning_tokens": s.reasoning_tokens,
                "prompt_chars": s.prompt_chars,
                "response_chars": s.response_chars,
                "finish_reason": usage.get("finish_reason"),
                "gen_params": s.gen_params or {},
                "diagnoses": analysis.get("diagnoses") or [],
                "is_suspicious": analysis.get("is_suspicious", False),
                "abnormal_notes": analysis.get("abnormal_notes") or [],
                "is_abnormal": analysis.get("is_abnormal", False),
                "lang_notes": analysis.get("lang_notes") or [],
                # IFEval 逐条约束明细 (RULE 类): [{type, args, satisfied}]
                "ifeval": ifeval,
            }
            if lite:
                # lite: ifeval 只留汇总 (satisfied/total/rate), 不含逐条 details (展开时按需拉)
                row["ifeval"] = {k: v for k, v in ifeval.items() if k != "details"}
            flat_reqs.append(row)
    # 各评测集汇总信息 (供明细视图顶部展示该集分数/乱码/性能)
    bench_infos = {}
    for r in run_results:
        meta = r.benchmark_meta
        a = r.aggregate
        g = a.get("gibberish", {}) or {}
        bench_infos[r.benchmark] = {
            "display": meta.display_name,
            "description": meta.description,
            "stage": meta.stage.value,
            "task_type": meta.task_type.value,
            "num_samples": r.num_samples,
            "score_label": _score_label(r),
            "score_value": _score_value(r),
            "gibberish": {
                "suspicious_rate": g.get("suspicious_rate", 0),
                "suspicious_count": g.get("suspicious_count", 0),
                "abnormal_count": g.get("abnormal_count", 0),
                "overall_grade": g.get("overall_grade", "-"),
            },
        }
    # 数据集列表 (供筛选下拉)
    bench_options = []
    seen = set()
    for r in run_results:
        if r.benchmark not in seen:
            seen.add(r.benchmark)
            bench_options.append({"internal": r.benchmark, "display": r.benchmark_meta.display_name})
    return flat_reqs, bench_infos, bench_options


def _render_html(run_results: List[RunResult]) -> str:
    model_name = run_results[0].model_name if run_results else "-"
    started = min((r.started_at for r in run_results), default="-")
    finished = max((r.finished_at for r in run_results), default="-")

    # 汇总卡片: 平均分 + 总乱码率 + 评测集数 + 可疑样本数
    scores = [_score_value(r) for r in run_results]
    scores = [s for s in scores if s is not None]
    avg_score = round(sum(scores) / len(scores), 2) if scores else None
    total_suspicious = sum(r.aggregate.get("gibberish", {}).get("suspicious_count", 0) for r in run_results)
    total_samples = sum(r.num_samples for r in run_results)
    overall_gib_rate = round(total_suspicious / total_samples * 100, 2) if total_samples else 0

    # 伪流式提醒横幅 (任一评测集检测到伪流式则显示)
    any_fake = any(getattr(r, "fake_stream", False) for r in run_results)
    if any_fake:
        fake_stream_banner = (
            '<div class="warn-banner">⚠ <b>检测到伪流式</b>: 该服务虽支持流式(stream=true), '
            '但实际是先把响应推理完再批量吐出(首字延迟=完整推理时间, 生成阶段≈0)。'
            '因此 <b>TPOT/生成阶段无意义, 已自动退化为非流式口径</b>(TTFT 用端到端近似, 标注≈)。'
            '如需真实 TPOT, 请检查服务侧(如 LiteLLM/Nginx 代理)是否正确透传流式 '
            '(关闭 response buffering / 启用 stream passthrough)。</div>'
        )
    else:
        fake_stream_banner = ""

    # 总览卡: 平均分 / 总请求数(正确错误) / 平均输入tokens / 平均输出tokens(含思维链) / 总耗时 / 整体乱码率
    all_results = [s for r in run_results for s in r.results]
    correct_cnt = sum(1 for s in all_results if s.correct is True)
    wrong_cnt = sum(1 for s in all_results if s.correct is False)
    # 平均输入/输出 token (跨所有请求)
    pt_list = [s.prompt_tokens for s in all_results if s.prompt_tokens is not None]
    ct_list = [s.completion_tokens for s in all_results if s.completion_tokens is not None]
    avg_in_tok = round(sum(pt_list) / len(pt_list)) if pt_list else None
    avg_out_tok = round(sum(ct_list) / len(ct_list)) if ct_list else None
    # 整个任务执行总时间 (最早 started_at → 最晚 finished_at)
    total_dur_str = _fmt_duration(started, finished)
    summary_cards = "\n".join([
        _card("平均分", f"{avg_score}" if avg_score is not None else "-", "跨所有可打分评测集"),
        _card("总请求数", str(len(all_results)), f"正确 {correct_cnt} / 错误 {wrong_cnt}"),
        _card("平均输入tokens", f"{avg_in_tok}" if avg_in_tok is not None else "-", "每条请求 prompt"),
        _card("平均输出tokens", f"{avg_out_tok}" if avg_out_tok is not None else "-", "含思维链"),
        _card("总耗时", total_dur_str, f"{len(run_results)} 个评测集"),
        _card("整体乱码率", f"{overall_gib_rate}%", f"{total_suspicious} 条可疑"),
    ])

    # 是否真流式 (TTFT/TPOT/gen_time 有意义性: 仅真流式才显示, 非流式/伪流式都不显示)
    is_real_stream = any(s.real_stream for s in all_results if s.real_stream is not None)

    # —— 评测集明细 (合并原"评测集明细"+"性能详情"为一个表) ——
    def cell(d, *keys, fmt="{:.1f}"):
        """取 dist 统计里 mean/p50 中先有的值"""
        v = None
        for k in keys:
            if d and d.get(k) is not None:
                v = d[k]; break
        if v is None or v == 0 and d.get("count", 0) == 0:
            return "-"
        try: return fmt.format(float(v))
        except (ValueError, TypeError): return str(v)

    rows = []
    for r in run_results:
        meta = r.benchmark_meta
        a = r.aggregate
        g = a.get("gibberish", {})
        gib_rate = g.get("suspicious_rate", 0)
        gib_pct = f"{gib_rate*100:.1f}%" if isinstance(gib_rate, (int, float)) else "-"
        grade = g.get("overall_grade", "-")
        grade_cls = "good" if grade.startswith(("A", "B")) else ("warn" if grade.startswith("C") else "bad")
        sv = _score_value(r)
        bar_pct = sv if sv is not None else 0
        stage_tag = f'<span class="tag {meta.stage.value}">{meta.stage.value}</span>'
        # 性能指标
        tps = a.get("tokens_per_sec", {}) or {}
        lat = a.get("latency", {}) or {}
        ttft = a.get("ttft_ms", {}) or {}
        tu = a.get("token_usage", {}) or {}
        lc = a.get("length_chars", {}) or {}
        tpot = a.get("tpot_ms", {}) or {}
        gent = a.get("gen_time_ms", {}) or {}
        pt = tu.get("prompt", {}) or {}
        ct = tu.get("completion", {}) or {}
        rt_tok = tu.get("reasoning", {}) or {}
        pch = lc.get("prompt", {}) or {}
        rch = lc.get("response", {}) or {}
        rch_lc = lc.get("reasoning", {}) or {}
        # TTFT/TPOT/生成阶段: 仅真流式才显示
        stream_cells = ""
        if is_real_stream:
            stream_cells = (f"<td>{cell(ttft,'p50')}</td>"
                            f"<td>{cell(tpot,'p50','mean')}</td>"
                            f"<td>{cell(gent,'p50','mean')}</td>")
        # token: 入/出/思 ; 字符: 入/出/思
        tok_str = f"{cell(pt,'mean')}/{cell(ct,'mean')}" + (f"/{cell(rt_tok,'mean')}" if rt_tok else "/-")
        char_str = f"{cell(pch,'mean','p50')}/{cell(rch,'mean','p50')}" + (f"/{cell(rch_lc,'mean','p50')}" if rch_lc else "/-")
        rows.append(f"""<tr>
          <td><a class="bench-link" onclick="showBench('{r.benchmark}')" title="查看该集每条请求明细"><b>{escape(meta.display_name)}</b></a><br><span style="color:var(--muted);font-size:11px">{escape(meta.description[:40])}</span></td>
          <td>{stage_tag}</td>
          <td>{meta.task_type.value}</td>
          <td>{r.num_samples}</td>
          <td class="score">{_score_label(r)}<div class="bar"><div style="width:{bar_pct}%"></div></div></td>
          <td>{gib_pct}</td>
          <td><span class="badge {grade_cls}">{escape(grade)}</span></td>
          <td>{cell(tps,'mean','p50')}</td>
          <td>{cell(lat,'p50')}/{cell(lat,'p95')}</td>
          {stream_cells}
          <td class="rpt-mono">{tok_str}</td>
          <td class="rpt-mono">{char_str}</td>
        </tr>""")
    # 表头 & colspan 随是否真流式变化
    base_header = "<th>评测集</th><th>阶段</th><th>类型</th><th>样本数</th><th>分数</th><th>乱码率</th><th>健康度</th><th>速度 tok/s</th><th>延迟 p50/p95 ms</th>"
    if is_real_stream:
        bench_header = base_header + "<th>TTFT p50 ms</th><th>TPOT p50 ms</th><th>生成阶段 p50 ms</th><th>tokens(入/出/思)</th><th>字符(入/出/思)</th>"
        perf_extra_title = " / TTFT / TPOT"
        bench_colspan = 15
    else:
        bench_header = base_header + "<th>tokens(入/出/思)</th><th>字符(入/出/思)</th>"
        perf_extra_title = ""
        bench_colspan = 12

    # —— 全部数据集的平均/汇总行 ——
    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None
    def _pct_str(rate):
        return f"{rate*100:.1f}%" if isinstance(rate, (int, float)) else "-"
    # 跨所有请求的分布 (用于平均行的速度/延迟/token/字符)
    all_tps = [s.tokens_per_sec for s in all_results if s.tokens_per_sec is not None]
    all_lat = [s.latency_ms for s in all_results if s.latency_ms is not None]
    all_pt = [s.prompt_tokens for s in all_results if s.prompt_tokens is not None]
    all_ct = [s.completion_tokens for s in all_results if s.completion_tokens is not None]
    all_rt_tok = [s.reasoning_tokens for s in all_results if s.reasoning_tokens is not None]
    all_pch = [s.prompt_chars for s in all_results if s.prompt_chars is not None]
    all_rch = [s.response_chars for s in all_results if s.response_chars is not None]
    all_rch_lc = [len(s.reasoning_content or "") for s in all_results if s.reasoning_content]
    avg_stream_cells = ""
    if is_real_stream:
        all_ttft = [s.ttft_ms for s in all_results if s.ttft_ms is not None]
        all_tpot = [s.tpot_ms for s in all_results if s.tpot_ms is not None]
        all_gent = [s.gen_time_ms for s in all_results if s.gen_time_ms is not None]
        avg_stream_cells = (f"<td>{_avg(all_ttft) or '-'}</td>"
                            f"<td>{_avg(all_tpot) or '-'}</td>"
                            f"<td>{_avg(all_gent) or '-'}</td>")
    sorted_lat = sorted(all_lat)
    lat_p50 = round(sorted_lat[len(sorted_lat)//2], 1) if sorted_lat else "-"
    lat_p95 = round(sorted_lat[min(int(len(sorted_lat)*0.95), len(sorted_lat)-1)], 1) if sorted_lat else "-"
    avg_tok = f"{_avg(all_pt) or '-'}/{_avg(all_ct) or '-'}" + (f"/{_avg(all_rt_tok)}" if all_rt_tok else "/-")
    avg_char = f"{_avg(all_pch) or '-'}/{_avg(all_rch) or '-'}" + (f"/{_avg(all_rch_lc)}" if all_rch_lc else "/-")
    # 平均分
    scores_list = [_score_value(r) for r in run_results]
    scores_list = [s for s in scores_list if s is not None]
    avg_score_str = f"{round(sum(scores_list)/len(scores_list), 1):g}" if scores_list else "-"
    # 总体乱码率
    overall_gib = sum(r.aggregate.get("gibberish", {}).get("suspicious_rate", 0) for r in run_results) / len(run_results) if run_results else 0
    avg_row = f"""<tr style="background:rgba(79,70,229,.06);font-weight:600">
      <td>📊 全部平均</td><td>-</td><td>-</td>
      <td>{total_samples}</td>
      <td class="score">{avg_score_str}</td>
      <td>{_pct_str(overall_gib)}</td>
      <td>-</td>
      <td>{_avg(all_tps) or '-'}</td>
      <td>{lat_p50}/{lat_p95}</td>
      {avg_stream_cells}
      <td class="rpt-mono">{avg_tok}</td>
      <td class="rpt-mono">{avg_char}</td>
    </tr>"""
    bench_rows = ("\n".join(rows) + avg_row) if rows else f'<tr><td colspan="{bench_colspan}" style="color:var(--muted)">无结果</td></tr>'

    # 每条请求明细 (交互式表格, 含搜索/筛选, 复刻请求浏览器样式)
    # 把所有请求摊平成 JSON, 嵌入页面, 由前端 JS 渲染表格 + 展开详情
    flat_reqs, bench_infos, bench_options = build_request_data(run_results)
    req_data_json = json.dumps(flat_reqs, ensure_ascii=False).replace("</", "<\\/")
    bench_options_json = json.dumps(bench_options, ensure_ascii=False).replace("</", "<\\/")
    bench_infos_json = json.dumps(bench_infos, ensure_ascii=False).replace("</", "<\\/")
    # request_details 现在是一个容器 + 嵌入的 JSON, 真正的表格由 JS 渲染
    # 工具栏+表格+分页包在 #rpt-host 里, initRptTable 重建其 innerHTML
    request_details = (
        '<div id="rpt-host">'
        '<div class="req-toolbar">'
        '<input type="text" id="rpt_search" placeholder="🔍 搜索 hash / 题目 / 回答 / 思维链 / 关键词..." oninput="rptReset()">'
        '<label>评测集: <select id="rpt_bench" onchange="rptReset()"><option value="">全部</option></select></label>'
        '<label>正确性: <select id="rpt_correct" onchange="rptReset()">'
        '<option value="">全部</option><option value="true">✓ 正确</option><option value="false">✗ 错误</option>'
        '<option value="error">⚠ 出错</option><option value="suspicious">⚠ 乱码/异常</option>'
        '<option value="reasoning">有思维链</option></select></label>'
        '<span id="rpt_count" style="color:var(--muted);font-size:12px"></span>'
        '<label>每页: <select id="rpt_pagesize" onchange="rptReset()"><option value="20" selected>20</option><option value="50">50</option><option value="100">100</option><option value="200">200</option><option value="999999">全部</option></select></label>'
        '</div>'
        '<table class="rpt-table"><thead><tr id="rpt_thead"></tr></thead><tbody id="rpt_tbody"></tbody></table>'
        '<div class="rpt-pager" id="rpt_pager"></div>'
        '</div>'
        f'<script>window.RPT_REQS = {req_data_json}; window.RPT_BENCHES = {bench_options_json}; window.RPT_BENCH_INFOS = {bench_infos_json};</script>'
    )

    # 乱码卡片 (每个评测集一张): 乱码率 + 输出异常率 (分开, 避免截断/空输出被算成乱码)
    gib_cards = []
    for r in run_results:
        g = r.aggregate.get("gibberish", {})
        if not g:
            continue
        diag_lines = "".join(
            f'<div style="font-size:12px;color:var(--muted)">{escape(k)}: {v}</div>'
            for k, v in g.get("diagnosis_counts", {}).items()
        )
        # 输出异常 (空输出/截断) 单列, 不混入乱码诊断
        abn = g.get("abnormal_counts", {}) or {}
        abn_lines = "".join(
            f'<div style="font-size:12px;color:var(--warn)">{escape(k)}: {v}</div>'
            for k, v in abn.items()
        )
        abn_cnt = g.get("abnormal_count", 0)
        abn_sub = f" · 输出异常 {abn_cnt}/{g.get('total_samples',0)}(截断/空)" if abn_cnt else ""
        gib_cards.append(_card(
            r.benchmark_meta.display_name,
            f"{g.get('suspicious_rate',0)*100:.1f}%",
            f"{g.get('suspicious_count',0)}/{g.get('total_samples',0)} 乱码 · {g.get('overall_grade','')}{abn_sub}",
            extra=diag_lines + abn_lines,
        ))
    gibberish_cards = "\n".join(gib_cards) if gib_cards else '<div class="card">无乱码数据</div>'

    # 可疑/异常样本示例 (乱码可疑 + 输出异常如截断/空)
    suspicious = []
    for r in run_results:
        for s in r.results:
            if not s.analysis:
                continue
            if s.analysis.get("is_suspicious") or s.analysis.get("is_abnormal"):
                suspicious.append((r.benchmark_meta.display_name, s))
    suspicious = suspicious[:5]
    if suspicious:
        samp_html = []
        for bench_name, s in suspicious:
            # 乱码诊断 + 输出异常诊断合并展示
            diags = "; ".join((s.analysis.get("diagnoses", []) or []) + (s.analysis.get("abnormal_notes", []) or []))
            tag = "乱码" if s.analysis.get("is_suspicious") else "异常"
            samp_html.append(f"""<div class="sample">
              <div><b>{escape(bench_name)}</b> · {escape(s.sample_id)} <span class="badge {'bad' if s.analysis.get('is_suspicious') else 'warn'}">{tag}</span></div>
              <div class="diag">诊断: {escape(diags)}</div>
              <pre>响应: {escape((s.response or '')[:500])}</pre>
            </div>""")
        suspicious_samples = "\n".join(samp_html)
    else:
        suspicious_samples = '<div class="card">未发现可疑输出 🎉</div>'

    html = HTML_TEMPLATE.format(
        model_name=escape(model_name),
        benchmarks_count=len(run_results),
        started=escape(str(started)[:19]),
        finished=escape(str(finished)[:19]),
        fake_stream_banner=fake_stream_banner,
        summary_cards=summary_cards,
        bench_header=bench_header,
        bench_rows=bench_rows,
        perf_extra_title=perf_extra_title,
        gibberish_cards=gibberish_cards,
        suspicious_samples=suspicious_samples,
        request_details=request_details,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    # 内联 rpt_table.js (与 SPA 任务详情"查看明细"页共用同一套表格渲染逻辑)。
    # 用占位符 __RPT_TABLE_JS__ 是因为 rpt_table.js 含大量 { } (JS 对象/函数体),
    # 不能作为 .format 的字段 (会与模板的 {{ }} 转义冲突), 故在 format 之后字符串替换。
    rpt_js_path = os.path.join(os.path.dirname(__file__), "server", "static", "rpt_table.js")
    try:
        with open(rpt_js_path, "r", encoding="utf-8") as f:
            rpt_js = f.read()
        html = html.replace("__RPT_TABLE_JS__", rpt_js)
    except OSError:
        html = html.replace("__RPT_TABLE_JS__", "/* rpt_table.js 加载失败 */")
    return html


def _fmt_duration(started: str, finished: str) -> str:
    """把 两个 ISO 时间字符串 之间的时长格式成 'X分Y秒' / 'X秒'。

    解析失败则返回 '-'。
    """
    try:
        fmts = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")
        s = _parse_dt(started, fmts)
        f = _parse_dt(finished, fmts)
        if s is None or f is None or f < s:
            return "-"
        secs = int((f - s).total_seconds())
        if secs < 60:
            return f"{secs}秒"
        m, sec = divmod(secs, 60)
        if m < 60:
            return f"{m}分{sec}秒"
        h, m = divmod(m, 60)
        return f"{h}时{m}分{sec}秒"
    except Exception:  # noqa: BLE001
        return "-"


def _parse_dt(s: str, fmts):
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _card(label: str, value: str, sub: str = "", extra: str = "") -> str:
    return f"""<div class="card">
      <div class="label">{escape(label)}</div>
      <div class="value">{escape(value)}</div>
      <div class="sub">{escape(sub)}</div>
      {extra}
    </div>"""
