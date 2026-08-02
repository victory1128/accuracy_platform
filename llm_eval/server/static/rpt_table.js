// 共享: 请求明细表格渲染器 (报告HTML + SPA任务详情"查看明细"页共用同一套逻辑)
// 报告页和SPA页都调用 initRptTable(container, {data, benches, benchInfos, opts}) 渲染出
// 与"完整报告每条请求明细"完全一致的表格 (TYPE_COLS 分类型列 / 搜索筛选 / 展开 / 分页)。
//
// container: 要挂载的 DOM 元素 (其 innerHTML 会被替换为工具栏+表格+分页)
// data:      flat_reqs 数组 (报告同款: 每条含 question/prompt/response/reasoning_content/
//            extracted/gold/correct/score/性能字段/analysis派生字段等)
// benches:   评测集下拉选项 [{internal, display}]
// benchInfos:{internal: {display, task_type, stage, num_samples, score_label, gibberish, ...}}
// opts:      { showToolbar(默认true), initialBench(默认''), onEnded(完成态无轮询回调) }
//
// 返回一个控制器对象 { refresh(newData), goto(bench) } 供调用方刷新/切换集。

function initRptTable(container, data, benches, benchInfos, opts) {
  opts = opts || {};
  const esc = (s) => (s == null ? '' : String(s)).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  let DATA = data || [];
  let BENCHES = benches || [];
  let BENCH_INFOS = benchInfos || {};
  let expanded = new Set();
  let PAGE = 1;
  const REAL_STREAM = (DATA || []).some(r => r.real_stream === true);
  // 轻量模式: DATA 不含大文本 (prompt/response/reasoning_content)。展开某行时按需拉完整明细。
  // fetchDetail(hash) → Promise<完整row>; 报告页数据完整, 不传 fetchDetail (直接展开)。
  const fetchDetail = opts.fetchDetail;
  const DETAIL_CACHE = new Map();   // hash → 完整 row (拉取后缓存, 避免重复请求)

  // ---- 工具栏 + 表格骨架 ----
  const showToolbar = opts.showToolbar !== false;
  container.innerHTML =
    (showToolbar ? `<div class="req-toolbar">
      <input type="text" id="rpt_search" placeholder="🔍 搜索 hash / 题目 / 回答 / 思维链 / 关键词..." oninput="rptReset()">
      <label>评测集: <select id="rpt_bench" onchange="rptReset()"><option value="">全部</option></select></label>
      <label>正确性: <select id="rpt_correct" onchange="rptReset()">
      <option value="">全部</option><option value="true">✓ 正确</option><option value="false">✗ 错误</option>
      <option value="error">⚠ 出错</option><option value="suspicious">⚠ 乱码/异常</option>
      <option value="reasoning">有思维链</option></select></label>
      <span id="rpt_count" style="color:var(--muted);font-size:12px"></span>
      <label>每页: <select id="rpt_pagesize" onchange="rptReset()"><option value="20" selected>20</option><option value="50">50</option><option value="100">100</option><option value="200">200</option><option value="999999">全部</option></select></label>
    </div>` : '') +
    '<table class="rpt-table"><thead><tr id="rpt_thead"></tr></thead><tbody id="rpt_tbody"></tbody></table>' +
    '<div class="rpt-pager" id="rpt_pager"></div>';

  function badge(r) {
    if (r.error) return '<span class="badge warn">出错</span>';
    if (r.correct === true) return '<span class="badge good">✓</span>';
    if (r.correct === false) return '<span class="badge bad">✗</span>';
    if (r.score != null) return '<span class="badge neutral">' + (typeof r.score === 'number' ? r.score.toFixed(1) : esc(r.score)) + '</span>';
    return '<span class="badge neutral">-</span>';
  }
  function flags(r) {
    let s = '';
    if ((r.diagnoses || []).length) s += ' <span class="badge warn" title="' + esc(r.diagnoses.join('; ')) + '">乱码</span>';
    if ((r.abnormal_notes || []).length) s += ' <span class="badge warn" style="background:rgba(251,191,36,.15)" title="' + esc(r.abnormal_notes.join('; ')) + '">异常</span>';
    if ((r.lang_notes || []).length) s += ' <span class="badge neutral" title="' + esc(r.lang_notes.join('; ')) + '">语言</span>';
    return s;
  }
  function filtered() {
    const q = (document.getElementById('rpt_search').value || '').trim().toLowerCase();
    const bf = document.getElementById('rpt_bench').value;
    const cf = document.getElementById('rpt_correct').value;
    return DATA.filter(r => {
      if (bf && r.benchmark !== bf) return false;
      if (cf === 'true' && r.correct !== true) return false;
      if (cf === 'false' && r.correct !== false) return false;
      if (cf === 'error' && !r.error) return false;
      if (cf === 'suspicious' && !r.is_suspicious && !r.is_abnormal) return false;
      if (cf === 'reasoning' && !r.has_reasoning) return false;
      if (q) {
        const hay = ((r.request_hash || '') + ' ' + (r.question || '') + ' ' + (r.response || '') + ' ' + (r.prompt || '') + ' ' + (r.extracted || '') + ' ' + (r.error || '') + ' ' + (r.reasoning_content || '') + ' ' + (r.reference || '')).toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }
  function tokCell(r) {
    const pi = r.prompt_tokens != null ? r.prompt_tokens : '-';
    const coTotal = r.completion_tokens;
    const rt = r.reasoning_tokens;
    let coOut, rtStr;
    if (coTotal == null) { coOut = '-'; rtStr = ''; }
    else if (rt != null) {
      coOut = (rt >= coTotal) ? coTotal : Math.max(0, coTotal - rt);
      rtStr = '/<span style="color:#7c3aed" title="思维链token">' + rt + '</span>';
    } else { coOut = coTotal; rtStr = '/<span style="color:var(--muted)">-</span>'; }
    return pi + '/' + coOut + rtStr;
  }
  function tokPi(r) { return r.prompt_tokens != null ? r.prompt_tokens : '-'; }
  function tokRt(r) { return r.reasoning_tokens != null ? r.reasoning_tokens : '-'; }
  function tokCo(r) {
    if (r.completion_tokens == null) return '-';
    const rt = r.reasoning_tokens;
    if (rt != null) return (rt >= r.completion_tokens) ? r.completion_tokens : Math.max(0, r.completion_tokens - rt);
    return r.completion_tokens;
  }

  function perfHeaders() {
    let h = '<th>速度(tok/s)</th><th>延迟(ms)</th>';
    if (REAL_STREAM) h += '<th>TTFT(ms)</th><th>TPOT(ms)</th>';
    h += '<th>tokens(入/出/思)</th><th>字符(入/出/思)</th>';
    return h;
  }
  function perfCells(r) {
    let h = '<td>' + (r.tokens_per_sec != null ? r.tokens_per_sec : '-') + '</td>'
      + '<td>' + (r.latency_ms != null ? Math.round(r.latency_ms) : '-') + '</td>';
    if (REAL_STREAM) h += '<td>' + (r.ttft_ms != null ? Math.round(r.ttft_ms) : '-') + '</td><td>' + (r.tpot_ms ?? '-') + '</td>';
    h += '<td class="rpt-mono">' + tokCell(r) + '</td>'
      + '<td class="rpt-mono">' + (r.prompt_chars ?? '-') + '/' + (r.response_chars ?? '-') + (r.reasoning_chars ? '/<span style="color:#7c3aed">' + r.reasoning_chars + '</span>' : '') + '</td>';
    return h;
  }
  function perfCount() { return REAL_STREAM ? 6 : 4; }

  const TYPE_COLS = {
    mcq: {
      coreH: '<th>题面(节选)</th><th>标准答案</th><th>模型选择</th><th>结果</th>',
      coreC: (r) => '<td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(r.question) + '">' + esc((r.question || '').slice(0, 70)) + '</td>'
        + '<td class="rpt-mono">' + esc(r.gold || '-') + '</td>'
        + '<td class="rpt-mono">' + esc(r.extracted || '-') + '</td>'
        + '<td>' + badge(r) + flags(r) + '</td>',
      coreN: 4,
      detail: (r) => '<h5>模型选择 / 标准答案</h5><pre>选择: ' + esc(r.extracted || '(无)') + '\n标准: ' + esc(r.gold || '(无)') + '</pre>',
    },
    gen: {
      coreH: '<th>题面(节选)</th><th>标准答案</th><th>抽取答案</th><th>结果</th>',
      coreC: (r) => '<td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(r.question) + '">' + esc((r.question || '').slice(0, 70)) + '</td>'
        + '<td class="rpt-mono">' + esc(r.gold || '-') + '</td>'
        + '<td class="rpt-mono" title="' + esc(r.extracted) + '">' + esc((r.extracted || '-').slice(0, 40)) + '</td>'
        + '<td>' + badge(r) + flags(r) + '</td>',
      coreN: 4,
      detail: (r) => '<h5>抽取答案 / 标准答案</h5><pre>抽取: ' + esc(r.extracted || '(无)') + '\n标准: ' + esc(r.gold || '(无)') + '</pre>',
    },
    code: {
      coreH: '<th>题面(函数签名)</th><th>执行结果</th><th>错误信息</th>',
      coreC: (r) => '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(r.question) + '">' + esc((r.question || '').slice(0, 50)) + '</td>'
        + '<td>' + (r.correct === true ? '<span class="badge good">✓ pass</span>' : (r.correct === false ? '<span class="badge bad">✗ fail</span>' : '<span class="badge neutral">-</span>')) + flags(r) + '</td>'
        + '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(r.error) + '">' + (r.error ? esc(r.error.slice(0, 60)) : '-') + '</td>',
      coreN: 3,
      detail: (r) => '<h5>生成代码</h5><pre>' + esc(r.extracted || '(无)') + '</pre>'
        + (r.error ? '<h5 style="color:var(--bad)">执行错误</h5><pre>' + esc(r.error) + '</pre>' : '')
        + '<h5>执行结果 / 标准答案</h5><pre>执行: ' + (r.correct === true ? 'pass' : (r.correct === false ? 'fail' : '-')) + '\n标准: ' + esc(r.gold || '(无, 用例通过即对)') + '</pre>',
    },
    judge: {
      coreH: '<th>题面(节选)</th><th>参考答案</th><th>裁判分</th>',
      coreC: (r) => '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(r.question) + '">' + esc((r.question || '').slice(0, 50)) + '</td>'
        + '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(r.reference) + '">' + esc((r.reference || '-').slice(0, 50)) + '</td>'
        + '<td><b>' + (r.score != null ? (typeof r.score === 'number' ? r.score.toFixed(1) : esc(r.score)) : '-') + '</b>/10' + flags(r) + '</td>',
      coreN: 3,
      detail: (r) => '<h5>参考答案</h5><pre>' + esc(r.reference || '(无)') + '</pre>'
        + '<h5>裁判评分</h5><pre>分数: ' + (r.score != null ? r.score : '-') + ' / 10</pre>',
    },
    rule: {
      coreH: '<th>题面(节选)</th><th>满足率</th><th>约束</th><th>结果</th>',
      coreC: (r) => {
        const ie = r.ifeval || {};
        const sat = ie.satisfied != null ? ie.satisfied : '-';
        const tot = ie.total != null ? ie.total : '-';
        const rate = ie.rate != null ? (ie.rate * 100).toFixed(0) + '%' : '-';
        return '<td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(r.question) + '">' + esc((r.question || '').slice(0, 70)) + '</td>'
          + '<td class="rpt-mono"><b>' + rate + '</b></td>'
          + '<td class="rpt-mono">' + sat + '/' + tot + '</td>'
          + '<td>' + badge(r) + flags(r) + '</td>';
      },
      coreN: 4,
      detail: (r) => {
        const ie = r.ifeval || {};
        const details = ie.details || [];
        if (!details.length) return '<h5>逐条约束明细</h5><pre>(无约束数据)</pre>';
        const rows = details.map(d => {
          const ok = d.satisfied;
          return '<tr><td class="' + (ok ? 'ok' : 'no') + '">' + (ok ? '✓' : '✗') + '</td><td class="rpt-mono">' + esc(d.type || '-') + '</td><td class="rpt-mono">' + esc(JSON.stringify(d.args || {})) + '</td></tr>';
        }).join('');
        return '<h5>逐条约束明细 (' + (ie.satisfied ?? '-') + '/' + (ie.total ?? '-') + ' 满足)</h5>'
          + '<table class="constraint-table"><thead><tr><th>结果</th><th>约束类型</th><th>参数</th></tr></thead><tbody>' + rows + '</tbody></table>';
      },
    },
    generic: {
      coreH: '<th>评测集</th><th>题面(节选)</th><th>抽取答案</th><th>标准答案</th><th>结果</th>',
      coreC: (r) => '<td>' + esc(r.benchmark_display) + '<br><span class="tag ' + r.stage + '">' + r.stage + '</span></td>'
        + '<td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(r.question) + '">' + esc((r.question || '').slice(0, 70)) + '</td>'
        + '<td class="rpt-mono" title="' + esc(r.extracted) + '">' + esc((r.extracted || '-').slice(0, 30)) + '</td>'
        + '<td class="rpt-mono">' + esc(r.gold || '-') + '</td>'
        + '<td>' + badge(r) + flags(r) + '</td>',
      coreN: 5,
      detail: (r) => '<h5>抽取答案 / 标准答案</h5><pre>抽取: ' + esc(r.extracted || '(无)') + '\n标准: ' + esc(r.gold || '(无)') + '</pre>',
    },
  };
  function getTaskType() {
    const bf = document.getElementById('rpt_bench').value;
    if (bf && BENCH_INFOS[bf]) return BENCH_INFOS[bf].task_type;
    return 'generic';
  }
  function colCount() { return 1 + (TYPE_COLS[getTaskType()] || TYPE_COLS.generic).coreN + perfCount(); }

  function row(r) {
    const tt = getTaskType();
    const tc = TYPE_COLS[tt] || TYPE_COLS.generic;
    const h = r.request_hash || '-';
    const isOpen = expanded.has(h);
    let html = '<tr>'
      + '<td><span class="rpt-hash" data-hash="' + esc(h) + '">' + esc(h) + '</span></td>'
      + tc.coreC(r)
      + perfCells(r)
      + '</tr>';
    if (isOpen) {
      // 轻量模式: 展开用已拉取的完整明细 (DETAIL_CACHE), 否则用 lite 行 (大文本为空)。
      const d = DETAIL_CACHE.get(h) || r;
      const loading = fetchDetail && !DETAIL_CACHE.has(h);
      if (loading) {
        html += '<tr class="rpt-detail"><td colspan="' + colCount() + '" class="rpt-detail-inner"><div class="muted">加载完整明细...</div></td></tr>';
        return html;
      }
      let kv = '<div class="rpt-kv">'
        + '<span>hash: <b class="rpt-mono">' + esc(h) + '</b></span>'
        + '<span>id: <b>' + esc(r.sample_id) + '</b></span>'
        + '<span>评测集: <b>' + esc(r.benchmark_display) + '</b> (' + esc(r.task_type) + ')</span>'
        + '<span>结果: <b>' + (r.correct === true ? '✓' : (r.correct === false ? '✗' : '-')) + '</b> 分数: <b>' + (r.score ?? '-') + '</b></span>'
        + '<span>finish: <b>' + esc(r.finish_reason || '-') + '</b></span>'
        + '<span>速度: <b>' + (r.tokens_per_sec ?? '-') + '</b> tok/s</span>'
        + '<span>延迟: <b>' + (r.latency_ms != null ? Math.round(r.latency_ms) : '-') + '</b>ms</span>'
        + (REAL_STREAM ? '<span>TTFT: <b>' + (r.ttft_ms != null ? Math.round(r.ttft_ms) : '-') + '</b>ms</span><span>TPOT: <b>' + (r.tpot_ms ?? '-') + '</b>ms</span><span>生成阶段: <b>' + (r.gen_time_ms != null ? Math.round(r.gen_time_ms) : '-') + '</b>ms</span>' : '')
        + '<span>tokens: <b>入' + tokPi(r) + '/出' + tokCo(r) + '/' + (r.has_reasoning ? '思' + tokRt(r) : '-') + '</b></span>'
        + '<span>长度: <b>入' + (r.prompt_chars ?? '-') + '/出' + (r.response_chars ?? '-') + (r.reasoning_chars ? '/思' + r.reasoning_chars : '') + '</b>字符</span>'
        + '<span>生成参数: <b class="rpt-mono">' + esc(JSON.stringify(r.gen_params || {})) + '</b></span>'
        + (r.error ? '<span style="color:var(--bad)">错误: ' + esc(r.error) + '</span>' : '')
        + ((r.diagnoses || []).length ? '<span style="color:var(--warn)">乱码诊断: ' + esc(r.diagnoses.join('; ')) + '</span>' : '')
        + ((r.abnormal_notes || []).length ? '<span style="color:var(--warn)">输出异常: ' + esc(r.abnormal_notes.join('; ')) + '</span>' : '')
        + ((r.lang_notes || []).length ? '<span style="color:var(--muted)">语言提示: ' + esc(r.lang_notes.join('; ')) + '</span>' : '')
        + '</div>';
      html += '<tr class="rpt-detail"><td colspan="' + colCount() + '" class="rpt-detail-inner">' + kv
        + (d.system_prompt ? '<h5>System</h5><pre>' + esc(d.system_prompt) + '</pre>' : '')
        + '<h5>输入 (Prompt)</h5><pre>' + esc(d.prompt || d.question || '') + '</pre>'
        + (d.reasoning_content ? '<h5 class="reason">🧠 思维链 (' + d.reasoning_chars + '字符)</h5><pre>' + esc(d.reasoning_content) + '</pre>' : '')
        + '<h5 class="out">输出 (Response) ' + (d.correct === true ? '✓' : (d.correct === false ? '✗' : '')) + '</h5><pre>' + esc(d.response || '(空)') + '</pre>'
        + tc.detail(d)
        + '</td></tr>';
    }
    return html;
  }
  function renderPager(total, totalPages, ps) {
    const el = document.getElementById('rpt_pager');
    if (!el || ps >= 999999 || total <= ps) { if (el) el.innerHTML = ''; return; }
    let h = '';
    h += '<button onclick="rptGoto(1)" ' + (PAGE <= 1 ? 'disabled' : '') + '>«</button>';
    h += '<button onclick="rptGoto(' + (PAGE - 1) + ')" ' + (PAGE <= 1 ? 'disabled' : '') + '>‹</button>';
    const pages = new Set([1, totalPages, PAGE]);
    for (let i = 2; i <= 5 && i < totalPages; i++) pages.add(i);
    for (let i = totalPages - 1; i > totalPages - 5 && i > 1; i--) pages.add(i);
    for (let i = PAGE - 2; i <= PAGE + 2; i++) { if (i > 1 && i < totalPages) pages.add(i); }
    const sorted = [...pages].sort((a, b) => a - b);
    let prev = 0;
    sorted.forEach(p => {
      if (p - prev > 1) h += '<span style="color:var(--muted)">…</span>';
      h += '<button class="' + (p === PAGE ? 'active' : '') + '" onclick="rptGoto(' + p + ')">' + p + '</button>';
      prev = p;
    });
    h += '<button onclick="rptGoto(' + (PAGE + 1) + ')" ' + (PAGE >= totalPages ? 'disabled' : '') + '>›</button>';
    h += '<button onclick="rptGoto(' + totalPages + ')" ' + (PAGE >= totalPages ? 'disabled' : '') + '>»</button>';
    h += '<span class="rpt-jump">跳至 <input type="number" id="rpt_jump" min="1" max="' + totalPages + '" value="' + PAGE + '"> / ' + totalPages + ' 页</span>';
    el.innerHTML = h;
    const ji = document.getElementById('rpt_jump');
    if (ji) ji.onkeydown = function (e) { if (e.key === 'Enter') rptGoto(parseInt(this.value) || 1); };
  }
  function rptRender() {
    const rows = filtered();
    const total = rows.length;
    const ps = parseInt(document.getElementById('rpt_pagesize').value) || 20;
    const totalPages = ps >= 999999 ? 1 : Math.max(1, Math.ceil(total / ps));
    if (PAGE > totalPages) PAGE = totalPages;
    if (PAGE < 1) PAGE = 1;
    const start = (PAGE - 1) * ps;
    const pageRows = ps >= 999999 ? rows : rows.slice(start, start + ps);
    document.getElementById('rpt_count').textContent = '匹配 ' + total + ' 条' + (ps < 999999 && total > ps ? ' · 第 ' + PAGE + '/' + totalPages + ' 页' : '');
    const tb = document.getElementById('rpt_tbody');
    const cc = colCount();
    tb.innerHTML = pageRows.length ? pageRows.map(row).join('') : '<tr><td colspan="' + cc + '" class="empty">无匹配请求</td></tr>';
    const tc = TYPE_COLS[getTaskType()] || TYPE_COLS.generic;
    document.getElementById('rpt_thead').innerHTML = '<th>Hash</th>' + tc.coreH + perfHeaders();
    renderPager(total, totalPages, ps);
  }
  function rptGoto(p) {
    PAGE = p; rptRender();
    const el = container;
    const top = (el ? el.getBoundingClientRect().top + window.scrollY : 0) - 80;
    window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
  }
  function rptReset() { PAGE = 1; rptRender(); }

  // 暴露给 inline onclick 用 (报告页历史用法); SPA 页共用同一全局名
  window.rptGoto = rptGoto;
  window.rptReset = rptReset;

  // 事件委托: 点击 hash 展开/折叠
  document.getElementById('rpt_tbody').addEventListener('click', function (e) {
    const el = e.target.closest('.rpt-hash');
    if (el) {
      const h = el.getAttribute('data-hash');
      if (expanded.has(h)) {
        expanded.delete(h);
        rptRender();
      } else {
        expanded.add(h);
        rptRender();   // 先渲染 (轻量模式会显示"加载中")
        // 轻量模式: 首次展开该行时按需拉完整明细 (含 prompt/response/reasoning)
        if (fetchDetail && !DETAIL_CACHE.has(h)) {
          const rowObj = DATA.find(x => x.request_hash === h);
          fetchDetail(h, rowObj).then(function (full) {
            if (full) DETAIL_CACHE.set(h, full);
            if (expanded.has(h)) rptRender();   // 仍在展开态才重渲染 (用户可能已折叠)
          }).catch(function () { /* 失败: 保留"加载中", 用户可重试 (折叠再展开) */ });
        }
      }
    }
  });
  // 初始化筛选下拉
  if (showToolbar) {
    const bs = document.getElementById('rpt_bench');
    BENCHES.forEach(b => { const o = document.createElement('option'); o.value = b.internal; o.textContent = b.display; bs.appendChild(o); });
    if (opts.initialBench) bs.value = opts.initialBench;
  }

  // 首次渲染 (填表头+表体+分页)。报告页 showBench 会再 goto(),但 SPA 明细页
  // initRptTable 后不再调用, 故必须在此渲染, 否则表格空白。
  rptRender();

  // 控制器: 调用方可刷新数据 (运行中追加新样本) / 切换集
  return {
    refresh(newData) {
      DATA = newData || [];
      rptRender();
    },
    goto(bench) {
      const bs = document.getElementById('rpt_bench');
      if (bs) bs.value = bench;
      PAGE = 1; rptRender();
    },
    render: rptRender,
  };
}
