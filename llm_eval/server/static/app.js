// 单页应用: hash 路由 + 原生 JS。无构建步骤。
// 路由: #/login  #/dashboard  #/tasks/new  #/tasks/:id  #/admin  #/dev

const api = {
  async req(path, opts = {}) {
    const r = await fetch(path, {
      headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
      credentials: 'same-origin',
      ...opts,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    // 401 拦截: 仅对"受保护接口"跳登录。认证接口(login/register)的 401/409
    // 是凭据错误, 应透传真实 detail (如"用户名或密码错误"), 不能改成"未登录"。
    if (r.status === 401 && !opts.raw) { location.hash = '#/login'; throw new Error('未登录'); }
    const txt = await r.text();
    let data; try { data = txt ? JSON.parse(txt) : {}; } catch { data = { detail: txt }; }
    if (!r.ok) throw new Error(data.detail || data.message || ('HTTP ' + r.status));
    return data;
  },
  me: () => api.req('/api/auth/me'),
  register: (u, p) => api.req('/api/auth/register', { method: 'POST', body: { username: u, password: p }, raw: true }),
  login: (u, p) => api.req('/api/auth/login', { method: 'POST', body: { username: u, password: p }, raw: true }),
  logout: () => api.req('/api/auth/logout', { method: 'POST' }),
  benchmarks: () => api.req('/api/benchmarks'),
  serverInfo: () => api.req('/api/server-info'),
  tasks: () => api.req('/api/tasks'),
  tasksPage: (params) => api.req('/api/tasks/page' + (params ? '?' + new URLSearchParams(params) : '')),
  task: (id) => api.req('/api/tasks/' + id),
  createTask: (b) => api.req('/api/tasks', { method: 'POST', body: b }),
  deleteTask: (id) => api.req('/api/tasks/' + id, { method: 'DELETE' }),
  cloneTask: (id) => api.req('/api/tasks/' + id + '/clone', { method: 'POST' }),
  cancelTask: (id) => api.req('/api/tasks/' + id + '/cancel', { method: 'POST' }),
  logs: (id, after) => api.req('/api/tasks/' + id + '/logs' + (after ? '?after_id=' + after : '')),
  samples: (id, benchmark, offset, limit) => api.req('/api/tasks/' + id + '/samples' + '?' + new URLSearchParams({ benchmark: benchmark || '', offset: offset || 0, limit: limit || 100 })),
  sample: (id, sid) => api.req('/api/tasks/' + id + '/samples/' + sid),
  requests: (id, benchmark) => api.req('/api/tasks/' + id + '/requests' + (benchmark ? '?benchmark=' + encodeURIComponent(benchmark) : '')),
  requestSample: (id, sid) => api.req('/api/tasks/' + id + '/requests/' + encodeURIComponent(sid)),
  dryRun: (b) => api.req('/api/dev/dry-run', { method: 'POST', body: b }),
  quickSample: (b) => api.req('/api/dev/quick-sample', { method: 'POST', body: b }),
  reloadBench: () => api.req('/api/dev/reload-benchmarks', { method: 'POST' }),
  health: () => api.req('/api/dev/health'),
  changePassword: (oldp, newp) => api.req('/api/auth/password', { method: 'POST', body: { old_password: oldp, new_password: newp }, raw: true }),
  adminUsers: () => api.req('/api/admin/users'),
  adminTasks: (params) => api.req('/api/admin/tasks' + (params ? '?' + new URLSearchParams(params) : '')),
  adminTasksPage: (params) => api.req('/api/admin/tasks/page' + (params ? '?' + new URLSearchParams(params) : '')),
  adminStats: () => api.req('/api/admin/stats'),
  setRole: (id, role) => api.req('/api/admin/users/' + id + '/role', { method: 'PUT', body: { role } }),
  setActive: (id, v) => api.req('/api/admin/users/' + id + '/active', { method: 'PUT', body: { is_active: v } }),
  resetPassword: (id, newp) => api.req('/api/admin/users/' + id + '/password', { method: 'POST', body: { new_password: newp } }),
  trash: () => api.req('/api/admin/trash'),
  restoreTask: (id) => api.req('/api/admin/trash/' + id + '/restore', { method: 'POST' }),
  hardDeleteTask: (id) => api.req('/api/admin/trash/' + id, { method: 'DELETE' }),
  emptyTrash: () => api.req('/api/admin/trash', { method: 'DELETE' }),
};

// 全局状态
const S = { user: null, benchmarks: [], serverInfo: {} };

const esc = (s) => (s == null ? '' : String(s)).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
// 把 summary 里的 score 字符串解析为 0-100 量纲数值 (供求平均)。
// 兼容历史格式: "45.0%" / "45.0/100" / "4.5/10" / "45.0" / "-"。
function scoreTo100(s) {
  if (s == null) return null;
  const str = String(s).trim();
  const m = str.match(/-?\d+(\.\d+)?/);
  if (!m) return null;
  let v = parseFloat(m[0]);
  // 仅 "/10" (后跟非数字或结尾) 才是 0-10 量纲; "/100" 会被此正则排除, 不误乘。
  if (/\/10(?!\d)/.test(str)) v *= 10;        // 0-10 量纲 -> 0-100
  // 含 % 或 /100 或无后缀: 都视作 0-100 量纲, 不换算
  return v;
}
// summary 平均分 (0-100 量纲, 1 位小数)。无可解析分数返回 null。
function avgScore(summary) {
  if (!summary || !summary.length) return null;
  const vals = summary.map(s => scoreTo100(s.score)).filter(v => v != null && !isNaN(v));
  if (!vals.length) return null;
  return (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1);
}
const badge = (status) => `<span class="badge ${status}">${({pending:'排队',running:'运行中',done:'完成',failed:'失败',cancelled:'已取消'})[status] || status}</span>`;
// 分页条: 页大小下拉 (20/50/100/200/全部) + « ‹ 1…n › » + 跳页输入。
// onGoto(page) / onPageSize(ps) 为全局函数名; pageSize=0 表示全部。
function renderPager(page, total, pageSize, onGoto, onPageSize) {
  if (total <= 0) return '';
  const ps = (pageSize === 0 || pageSize == null) ? 999999 : pageSize;
  const totalPages = ps >= 999999 ? 1 : Math.max(1, Math.ceil(total / ps));
  if (ps >= 999999) {
    // 全部: 只显示条数 + 页大小切换, 无翻页
    return `<div class="pager"><span class="pager-info">共 ${total} 条</span>${_pageSizeSel(pageSize, onPageSize)}</div>`;
  }
  if (total <= ps) return `<div class="pager"><span class="pager-info">共 ${total} 条</span>${_pageSizeSel(pageSize, onPageSize)}</div>`;
  page = Math.min(Math.max(1, page), totalPages);
  let h = '<div class="pager"><span class="pager-info">共 ' + total + ' 条 · 第 ' + page + '/' + totalPages + ' 页</span>';
  h += '<span class="pager-btns">';
  h += `<button class="pg-btn" onclick="${onGoto}(1)" ${page <= 1 ? 'disabled' : ''}>«</button>`;
  h += `<button class="pg-btn" onclick="${onGoto}(${page - 1})" ${page <= 1 ? 'disabled' : ''}>‹</button>`;
  const pages = new Set([1, totalPages, page]);
  for (let i = 2; i <= 5 && i < totalPages; i++) pages.add(i);
  for (let i = totalPages - 1; i > totalPages - 5 && i > 1; i--) pages.add(i);
  for (let i = page - 2; i <= page + 2; i++) { if (i > 1 && i < totalPages) pages.add(i); }
  const sorted = [...pages].sort((a, b) => a - b);
  let prev = 0;
  sorted.forEach(p => {
    if (p - prev > 1) h += '<span class="pg-ellipsis">…</span>';
    h += `<button class="pg-btn ${p === page ? 'active' : ''}" onclick="${onGoto}(${p})">${p}</button>`;
    prev = p;
  });
  h += `<button class="pg-btn" onclick="${onGoto}(${page + 1})" ${page >= totalPages ? 'disabled' : ''}>›</button>`;
  h += `<button class="pg-btn" onclick="${onGoto}(${totalPages})" ${page >= totalPages ? 'disabled' : ''}>»</button>`;
  h += '</span>';
  h += `<span class="pg-jump">跳至 <input type="number" min="1" max="${totalPages}" value="${page}" onkeydown="if(event.key==='Enter'){${onGoto}(parseInt(this.value)||1);}"> / ${totalPages} 页</span>`;
  h += _pageSizeSel(pageSize, onPageSize);
  h += '</div>';
  return h;
}
// 页大小下拉 (内部用)
function _pageSizeSel(pageSize, onPageSize) {
  const cur = (pageSize == null) ? 20 : pageSize;
  const opts = [20, 50, 100, 200, 0].map(v => {
    const label = v === 0 ? '全部' : String(v);
    const sel = (cur === 0 && v === 0) || (cur === v && v !== 0) ? 'selected' : '';
    return `<option value="${v}" ${sel}>${label}</option>`;
  }).join('');
  return `<span class="pg-size">每页 <select onchange="${onPageSize}(parseInt(this.value))">${opts}</select></span>`;
}
// 角色层级 (与后端 auth.ROLE_LEVEL 一致): super(3) > admin(2) > user(1)
const ROLE_LEVEL = { super: 3, admin: 2, user: 1 };
const roleLvl = (r) => ROLE_LEVEL[r] || 0;
const ROLE_LABEL = { super: '超级管理员', admin: '管理员', user: '普通用户' };
// 密码输入框: 带小眼睛切换明文/密文。id=输入框id, placeholder/val/extra 同 input。
function pwField(id, placeholder, val, extra = '') {
  return `<div class="pw-wrap">
    <input type="password" id="${id}" placeholder="${esc(placeholder)}" ${val != null ? `value="${esc(val)}"` : ''} autocomplete="off" ${extra}>
    <span class="pw-eye" onclick="togglePw('${id}', this)">👁</span>
  </div>`;
}
function togglePw(id, eye) {
  const el = document.getElementById(id);
  if (!el) return;
  if (el.type === 'password') { el.type = 'text'; eye.textContent = '🙈'; }
  else { el.type = 'password'; eye.textContent = '👁'; }
}

// ----------------------------- 路由 -----------------------------
async function route() {
  const hash = location.hash.slice(1) || '/';
  // 离开任何页面都先停掉后台实时通道 (SSE + 明细页轮询), 避免泄漏/访问已销毁 DOM。
  // renderTaskDetail/renderSamplePage 自身也会停 (切到彼此时), 这里兜底覆盖其它目标页。
  if (sseES) { sseES.close(); sseES = null; }
  _spStopPoll();
  // 刷新时 #main 可能还不存在 (index.html 只有加载占位), 先确保容器就位
  let main = document.getElementById('main');
  if (!main) {
    document.getElementById('app').innerHTML = '<div class="container" id="main"><div class="empty">加载中...</div></div>';
    main = document.getElementById('main');
  }
  try {
    if (!S.user && hash !== '/login') {
      try { S.user = await api.me(); } catch { S.user = null; location.hash = '#/login'; return; }
    }
    if (hash === '/login') return renderAuth();
    if (!S.user) { location.hash = '#/login'; return; }

    // 加载评测集目录 (懒)
    if (!S.benchmarks.length) S.benchmarks = await api.benchmarks().catch(() => []);
    if (!S.serverInfo.version) S.serverInfo = await api.serverInfo().catch(() => ({}));

    if (hash === '/dashboard') return renderDashboard();
    if (hash === '/tasks/new') return renderNewTask();
    if (hash.startsWith('/tasks/')) {
      const parts = hash.split('/');
      // /tasks/:id/samples/:bench → 单集每条请求明细 (独立页, 同报告样式)
      if (parts.length >= 5 && parts[3] === 'samples') return renderSamplePage(parts[2], decodeURIComponent(parts[4] || ''));
      return renderTaskDetail(parts[2]);
    }
    if (hash === '/admin') return roleLvl(S.user.role) >= roleLvl('admin') ? renderAdmin() : (main.innerHTML = '<div class="alert err">需要管理员权限</div>');
    if (hash === '/dev') return renderDev();
    if (hash === '/password') return renderChangePassword();
    if (hash === '/logout') { await api.logout(); S.user = null; location.hash = '#/login'; return; }
    location.hash = '#/dashboard';
  } catch (e) {
    main.innerHTML = `<div class="alert err">加载失败: ${esc(e.message)}</div>`;
  }
}

function renderTopbar() {
  const nav = [
    ['#/dashboard', '仪表板'], ['#/tasks/new', '新建任务'],
    ['#/dev', '开发模式'],
  ];
  if (S.user && roleLvl(S.user.role) >= roleLvl('admin')) nav.push(['#/admin', '管理后台']);
  const hash = location.hash.slice(1) || '/';
  const links = nav.map(([h, t]) => `<a href="${h}" class="${h === hash ? 'active' : ''}">${t}</a>`).join('');
  return `<div class="topbar">
    <div class="brand">精度测试平台</div>
    <div class="nav">${links}</div>
    <div class="user">${esc(S.user ? S.user.username : '')}
      ${S.user && S.user.role !== 'user' ? `<span class="role">${esc(S.user.role)}</span>` : ''}
      <a href="#/password" class="btn secondary" style="padding:5px 12px;font-size:12px">改密</a>
      <a href="#/logout" class="btn secondary" style="padding:5px 12px;font-size:12px">登出</a>
    </div></div>`;
}

function setMain(html) {
  document.getElementById('app').innerHTML = renderTopbar() + `<div class="container" id="main">${html}</div>`;
}

// ----------------------------- 认证页 -----------------------------
function renderAuth() {
  document.getElementById('app').innerHTML = `<div class="auth-wrap"><div class="auth-card card">
    <h1>精度测试平台</h1><div class="sub">登录或注册以开始评测</div>
    <div class="auth-tabs"><button id="tab-login" class="active" onclick="switchAuth('login')">登录</button><button id="tab-register" onclick="switchAuth('register')">注册</button></div>
    <div id="auth-err" class="alert err" style="display:none"></div>
    <label>用户名</label><input type="text" id="au" placeholder="3-32 字符">
    <label>密码</label>${pwField('ap', '至少 6 位', null, 'onkeydown="if(event.key===\'Enter\')doAuth()"')}
    <button class="btn" style="width:100%;margin-top:14px" id="auth-btn" onclick="doAuth()">登录</button>
  </div></div>`;
}
function switchAuth(mode) {
  const isLogin = mode === 'login';
  document.getElementById('tab-login').classList.toggle('active', isLogin);
  document.getElementById('tab-register').classList.toggle('active', !isLogin);
  document.getElementById('auth-btn').textContent = isLogin ? '登录' : '注册';
}
async function doAuth() {
  const u = document.getElementById('au').value.trim();
  const p = document.getElementById('ap').value;
  const isLogin = document.getElementById('tab-login').classList.contains('active');
  const err = document.getElementById('auth-err');
  err.style.display = 'none';
  try {
    S.user = isLogin ? await api.login(u, p) : await api.register(u, p);
    location.hash = '#/dashboard';
  } catch (e) {
    err.textContent = e.message; err.style.display = 'block';
  }
}

// ----------------------------- 修改密码 (用户自己) -----------------------------
function renderChangePassword() {
  setMain(`<h1>修改密码</h1><div class="sub">当前账号: ${esc(S.user ? S.user.username : '')}</div>
    <div class="card" style="max-width:420px">
      <div id="pw-err" class="alert err" style="display:none"></div>
      <div id="pw-ok" class="alert" style="display:none">✓ 密码已修改</div>
      <label>原密码</label>${pwField('pw_old', '当前密码', null)}
      <label>新密码 (至少6位)</label>${pwField('pw_new1', '新密码', null)}
      <label>再次输入新密码</label>${pwField('pw_new2', '再输一次新密码', null, 'onkeydown="if(event.key===\'Enter\')doChangePassword()"')}
      <button class="btn" style="margin-top:14px" onclick="doChangePassword()">确认修改</button>
      <a class="btn secondary" href="#/dashboard">返回</a>
    </div>`);
}
async function doChangePassword() {
  const oldp = document.getElementById('pw_old').value;
  const n1 = document.getElementById('pw_new1').value;
  const n2 = document.getElementById('pw_new2').value;
  const err = document.getElementById('pw-err'), ok = document.getElementById('pw-ok');
  err.style.display = 'none'; ok.style.display = 'none';
  if (!oldp) return (err.textContent = '请输入原密码', err.style.display = 'block');
  if (n1.length < 6) return (err.textContent = '新密码至少 6 位', err.style.display = 'block');
  if (n1 !== n2) return (err.textContent = '两次输入的新密码不一致', err.style.display = 'block');
  if (n1 === oldp) return (err.textContent = '新密码不能与原密码相同', err.style.display = 'block');
  try {
    await api.changePassword(oldp, n1);
    ok.style.display = 'block';
    document.getElementById('pw_old').value = '';
    document.getElementById('pw_new1').value = '';
    document.getElementById('pw_new2').value = '';
  } catch (e) { err.textContent = e.message; err.style.display = 'block'; }
}

// ----------------------------- 仪表板 -----------------------------
let dashPage = 1, dashPageSize = 20;
async function renderDashboard() {
  setMain(`<h1>评测任务</h1><div class="sub">查看你的历史评测任务</div>
    <div class="actions" style="margin-bottom:14px"><a class="btn" href="#/tasks/new">+ 新建评测任务</a></div>
    <div id="task-list"><div class="empty">加载中...</div></div>`);
  await loadDashboard();
}
async function loadDashboard() {
  const tb = document.getElementById('task-list');
  if (!tb) return;
  try {
    const data = await api.tasksPage({ page: dashPage, page_size: dashPageSize });
    if (!data.items.length) { tb.innerHTML = '<div class="empty">还没有任务。<a href="#/tasks/new">创建第一个</a></div>'; return; }
    tb.innerHTML = `<div class="table-wrap"><table><thead><tr><th>名称</th><th>模式</th><th>模型</th><th>评测集</th><th>状态</th><th>分数</th><th>创建时间</th><th>操作</th></tr></thead><tbody>
      ${data.items.map(t => `<tr>
        <td class="cell-wrap"><a href="#/tasks/${t.id}">${esc(t.name)}</a></td>
        <td class="muted">${esc(t.mode)}</td>
        <td class="muted">${esc((t.model_config && t.model_config.name) || '-')}</td>
        <td class="muted cell-bench">${esc((t.benchmarks || []).join(', '))}</td>
        <td class="cell-status">${badge(t.status)}${t.status === 'failed' && t.error ? `<br><span class="muted" style="font-size:11px" title="${esc(t.error)}">${esc(t.error.slice(0,40))}${t.error.length>40?'…':''}</span>` : ''}</td>
        <td>${t.status === 'failed' ? '<span class="badge bad">失败</span>' : (avgScore(t.summary) != null ? `<b style="font-size:15px">${avgScore(t.summary)}</b><div class="muted" style="font-size:11px">${t.summary.length} 集平均</div>` : '-')}</td>
        <td class="muted">${esc((t.created_at || '').slice(0, 19).replace('T', ' '))}</td>
        <td class="actions cell-actions"><a class="btn secondary" href="#/tasks/${t.id}">查看</a>
          ${(t.status === 'pending' || t.status === 'running') ? `<button class="btn secondary" onclick="stopTask(${t.id})">停止</button>` : ''}
          <button class="btn secondary" onclick="cloneTask(${t.id})">克隆</button>
          <button class="btn danger" onclick="delTask(${t.id}, '${esc(t.name)}')">删除</button></td>
      </tr>`).join('')}</tbody></table></div>`
      + renderPager(dashPage, data.total, dashPageSize, 'dashGoto', 'dashSetPageSize');
  } catch (e) { tb.innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
}
async function dashGoto(p) { dashPage = Math.max(1, parseInt(p) || 1); await loadDashboard(); }
async function dashSetPageSize(ps) { dashPageSize = parseInt(ps); dashPage = 1; await loadDashboard(); }

async function cloneTask(id) {
  try {
    const c = await api.cloneTask(id);
    // 把克隆数据存到 sessionStorage, 新建页读取预填
    sessionStorage.setItem('clone', JSON.stringify(c));
    location.hash = '#/tasks/new';
  } catch (e) { alert('克隆失败: ' + e.message); }
}
async function delTask(id, name) {
  if (!confirm(`删除任务 "${name}"? 将移入回收站, 30天后彻底清除。`)) return;
  try { await api.deleteTask(id); route(); } catch (e) { alert('删除失败: ' + e.message); }
}
async function stopTask(id) {
  if (!confirm(`停止任务 #${id}?`)) return;
  try { await api.cancelTask(id); route(); } catch (e) { alert('停止失败: ' + e.message); }
}

// ----------------------------- 新建任务 -----------------------------
function renderNewTask() {
  const clone = JSON.parse(sessionStorage.getItem('clone') || 'null');
  sessionStorage.removeItem('clone');
  const mc = (clone && clone.model_config) || { name: '', base_url: '', api_key: '', model: '', temperature: 0.0, max_tokens: 2048, extra: {} };
  const jc = (clone && clone.judge_config) || { name: '', base_url: '', api_key: '', model: '', max_tokens: 256 };
  const rp = clone ? clone : { benchmarks: [], limit: '', concurrency: '', streaming: false, debug: false, mode: 'real', name: '' };
  const checked = new Set(rp.benchmarks || []);
  const benches = S.benchmarks.map(b => `<label class="bench-item">
    <input type="checkbox" value="${b.name}" ${checked.has(b.name) ? 'checked' : ''}>
    <b>${esc(b.display_name)}</b><span class="tag ${b.stage}">${b.stage}</span>${b.needs_judge ? '<span class="judge">·需裁判</span>' : ''}<span class="tag" style="background:rgba(100,116,139,.15);color:var(--muted)" title="数据集总条数 (留空采样=跑这么多)">${b.num_samples ?? '?'}条</span>
    <div class="desc">${esc(b.description)}</div></label>`).join('');
  const codeEnabled = S.serverInfo.code_exec_enabled;
  const sandboxMode = S.serverInfo.code_exec_sandbox || 'disabled';

  setMain(`<h1>新建评测任务</h1><div class="sub">填写被测模型 API (仅本次内存, 不落库) + 勾选评测集</div>
    ${!codeEnabled ? '<div class="alert">⚠ 代码沙箱未启用 (disabled): HumanEval/MBPP/BigCodeBench/LiveCodeBench 等需执行模型代码的评测集暂不可提交, 需先在 config.yaml 设 code_exec.sandbox 为 docker 或 subprocess。</div>' : `<div class="alert" style="border-color:var(--good);background:rgba(74,222,128,.08);color:var(--good)">✓ 代码沙箱已启用 (${esc(sandboxMode)} 模式): 代码评测集将在${sandboxMode === 'docker' ? '一次性 docker 容器内(禁网/只读/资源上限)执行' : '子进程内执行'}, 所有用户均可提交。</div>`}
    <div class="card">
      <div class="card-title">任务信息</div>
      <div class="row">
        <div><label>任务名 (可留空)</label><input type="text" id="f_name" value="${esc(rp.name || '')}" placeholder="如 glm5.2 知识评测"></div>
        <div><label>运行模式</label><select id="f_mode">
          <option value="real" ${rp.mode==='real'?'selected':''}>正式评测 (调真实 API)</option>
          <option value="dry_run" ${rp.mode==='dry_run'?'selected':''}>dry-run (模拟, 不调 API)</option>
          <option value="quick" ${rp.mode==='quick'?'selected':''}>快速试跑 (小样本)</option>
        </select></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">被测模型 API</div>
      <div class="row">
        <div><label>显示名</label><input type="text" id="m_name" value="${esc(mc.name)}" placeholder="如 glm-5.2"></div>
        <div><label>实际模型名 (model)</label><input type="text" id="m_model" value="${esc(mc.model)}" placeholder="如 glm-5.2-fp8"></div>
      </div>
      <div class="row">
        <div><label>API Base URL</label><input type="text" id="m_base" value="${esc(mc.base_url)}" placeholder="https://api.deepseek.com/v1"></div>
        <div><label>API Key 🔒 (仅内存)</label><input type="password" id="m_key" value="${esc(mc.api_key || '')}" placeholder="sk-..." autocomplete="off"></div>
      </div>
      <div class="row">
        <div><label>max_tokens (留空=用各评测集默认值, 如 IFEval 8192)</label><input type="number" id="m_maxtok" value="${mc.max_tokens ?? ''}" placeholder="如 4096, 留空则各评测集自带"></div>
        <div><label>temperature (留空=用各评测集默认值)</label><input type="number" step="0.1" id="m_temp" value="${mc.temperature ?? ''}" placeholder="如 0.0, 留空则各评测集自带"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">裁判模型 (可选, MT-Bench/AlpacaEval/Arena-Hard 需要)</div>
      <label><input type="checkbox" id="use_judge" ${clone && clone.judge_config ? 'checked' : ''} onchange="document.getElementById('judge_box').style.display=this.checked?'block':'none'" style="width:auto"> 使用裁判模型</label>
      <div id="judge_box" style="display:${clone && clone.judge_config ? 'block' : 'none'};margin-top:8px">
        <div class="row">
          <div><label>裁判显示名</label><input type="text" id="j_name" value="${esc(jc.name)}" placeholder="gpt-4o-judge"></div>
          <div><label>裁判模型名</label><input type="text" id="j_model" value="${esc(jc.model)}" placeholder="gpt-4o"></div>
        </div>
        <div class="row">
          <div><label>裁判 Base URL</label><input type="text" id="j_base" value="${esc(jc.base_url)}" placeholder="https://api.openai.com/v1"></div>
          <div><label>裁判 API Key 🔒</label><input type="password" id="j_key" value="${esc(jc.api_key || '')}" placeholder="sk-..." autocomplete="off"></div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">运行参数</div>
      <div class="row">
        <div><label>每集采样条数 (留空=全量, 即跑各评测集卡片显示的总条数)</label><input type="number" id="f_limit" value="${rp.limit ?? ''}" placeholder="如 5, 留空=全量"></div>
        <div><label>并发数</label><input type="number" id="f_conc" value="${rp.concurrency ?? ''}" placeholder="默认 4"></div>
      </div>
      <div class="row" style="margin-top:6px">
        <label style="display:inline"><input type="checkbox" id="f_stream" ${rp.streaming ? 'checked' : ''} style="width:auto"> 流式调用 (统计真实 TTFT/TPOT)</label>
        <label style="display:inline;margin-left:16px"><input type="checkbox" id="f_debug" ${rp.debug ? 'checked' : ''} style="width:auto"> 调试日志</label>
      </div>
    </div>
    <div class="card">
      <div class="card-title">评测集</div>
      <div style="margin-bottom:8px"><button class="btn secondary" onclick="document.querySelectorAll('.bench-item input').forEach(c=>c.checked=true)">全选</button>
        <button class="btn secondary" onclick="document.querySelectorAll('.bench-item input').forEach(c=>c.checked=false)">清空</button></div>
      <div class="bench-grid">${benches}</div>
    </div>
    <button class="btn" onclick="submitTask()">提交评测任务</button>
    <a class="btn secondary" href="#/dashboard">取消</a>`);
}

async function submitTask() {
  const benches = [...document.querySelectorAll('.bench-item input:checked')].map(c => c.value);
  if (!benches.length) return alert('请至少选择一个评测集');
  const mode = document.getElementById('f_mode').value;
  const body = {
    name: document.getElementById('f_name').value.trim(),
    mode,
    model_config: {
      name: document.getElementById('m_name').value.trim(),
      base_url: document.getElementById('m_base').value.trim(),
      api_key: document.getElementById('m_key').value,
      model: document.getElementById('m_model').value.trim(),
      // 留空=null -> 后端用各评测集自带 max_tokens/temperature
      max_tokens: document.getElementById('m_maxtok').value ? parseInt(document.getElementById('m_maxtok').value) : null,
      temperature: document.getElementById('m_temp').value !== '' ? parseFloat(document.getElementById('m_temp').value) : null,
    },
    use_judge: document.getElementById('use_judge').checked,
    judge_config: document.getElementById('use_judge').checked ? {
      name: document.getElementById('j_name').value.trim(),
      base_url: document.getElementById('j_base').value.trim(),
      api_key: document.getElementById('j_key').value,
      model: document.getElementById('j_model').value.trim(),
    } : null,
    benchmarks: benches,
    limit: document.getElementById('f_limit').value ? parseInt(document.getElementById('f_limit').value) : null,
    concurrency: document.getElementById('f_conc').value ? parseInt(document.getElementById('f_conc').value) : null,
    streaming: document.getElementById('f_stream').checked,
    debug: document.getElementById('f_debug').checked,
  };
  try {
    const t = await api.createTask(body);
    location.hash = '#/tasks/' + t.id;
  } catch (e) { alert('提交失败: ' + e.message); }
}

// ----------------------------- 任务详情 (SSE 实时) -----------------------------
let sseES = null;
// 任务详情实时样本查看状态: 当前展开的集名 + 是否在拉取中
let _tdTaskId = null, _tdViewBench = null, _tdLive = false, _tdLastProg = null;
async function renderTaskDetail(id) {
  if (sseES) { sseES.close(); sseES = null; }
  _spStopPoll();
  _tdTaskId = id; _tdViewBench = null; _tdLive = false;
  setMain(`<h1>任务详情</h1><div id="task-detail"><div class="empty">加载中...</div></div>`);
  let t;
  try { t = await api.task(id); } catch (e) { document.getElementById('task-detail').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; return; }
  const active = t.status === 'pending' || t.status === 'running';
  _tdLive = active;
  document.getElementById('task-detail').innerHTML = `
    <div class="sub" id="t-sub">${esc(t.name)} · <span id="t-badge">${badge(t.status)}</span> · 模型 ${esc((t.model_config && t.model_config.name) || '-')} · 评测集 ${esc((t.benchmarks || []).join(', '))}</div>
    <div class="card">
      <div>总进度: <span id="t-prog">-</span></div>
      <div class="bar"><div id="t-bar" style="width:0%"></div></div>
      <div id="t-bench-prog" style="margin-top:12px"></div>
      <div class="actions" style="margin-top:10px" id="t-actions">
        ${active ? `<button class="btn danger" onclick="cancelTask(${id})">取消任务</button>` : ''}
        ${t.report_path ? `<a class="btn" href="/api/tasks/${id}/report" target="_blank">查看完整报告 (HTML)</a>` : ''}
        <button class="btn secondary" onclick="cloneTask(${id})">克隆修改</button>
        <a class="btn secondary" href="#/dashboard">返回列表</a>
      </div>
    </div>
    <div id="t-samples"></div>
    <div id="t-summary"></div>
    <div id="t-error">${t.error ? `<div class="alert err"><b>错误:</b> ${esc(t.error)}</div>` : ''}</div>
    <div class="card"><div class="card-title">运行日志</div><div class="log" id="t-log"></div></div>`;

  // 初始渲染进度 (从 task.progress; 已完成任务也显示100%)
  renderBenchProgress(t.progress);

  // 若已是终态, 直接渲染摘要/报告链接 + 拉历史日志 (无需 SSE)
  if (!active) { renderTaskFinal(id, t); loadTaskLogs(id); }

  // SSE 订阅实时日志/进度 (仅活跃任务才连, 避免已完成任务反复重连闪烁)
  if (active) {
    sseES = new EventSource(`/api/tasks/${id}/events`);
    sseES.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      if (d.type === 'log') appendLog(d.level, d.message, d.ts);
      else if (d.type === 'progress') {
        // 评测集级进度: done/total 个评测集
        const p = document.getElementById('t-prog'); if (p) p.textContent = `${d.done}/${d.total}${d.current ? ' · ' + d.current : ''}`;
        const b = document.getElementById('t-bar'); if (b) b.style.width = (d.total ? d.done / d.total * 100 : 0) + '%';
      } else if (d.type === 'bench_progress') {
        // 各评测集进度: 展示每个数据集的进度条 + 已完成集的评分卡/查看明细按钮
        renderBenchProgress(d.progress);
      } else if (d.type === 'sample_result') {
        // 单条样本完成 (节流推送): 更新该集已完成数, 刷新"查看明细 (N)"按钮
        if (_tdLastProg && _tdLastProg[d.benchmark]) {
          _tdLastProg[d.benchmark].done = d.done;
          renderBenchProgress(_tdLastProg);
        }
      } else if (d.type === 'status') {
        if (d.error) appendLog('error', d.error, '');
        if (['done', 'failed', 'cancelled'].includes(d.status)) {
          // 终态: 关闭 SSE, 原地更新徽章+摘要+报告链接 (不重新渲染整页, 避免闪动)
          if (sseES) { sseES.close(); sseES = null; }
          _tdLive = false;   // 实时通道关闭; 但点数据集名字仍可进明细页 (/requests 走完整 JSON 结果)
          const bg = document.getElementById('t-badge'); if (bg) bg.innerHTML = badge(d.status);
          const act = document.getElementById('t-actions');
          if (act) act.innerHTML = `${d.report_path ? `<a class="btn" href="/api/tasks/${id}/report" target="_blank">查看完整报告 (HTML)</a>` : ''}<button class="btn secondary" onclick="cloneTask(${id})">克隆修改</button><a class="btn secondary" href="#/dashboard">返回列表</a>`;
          // 重新渲染进度区: 数据集名字仍可点 (已完成集走 /requests 完整结果, 不再提示去完整报告)
          renderBenchProgress(_tdLastProg);
          const sp = document.getElementById('t-samples');
          if (sp) sp.innerHTML = '';
          // 拉一次最新数据填摘要/错误
          api.task(id).then(ft => renderTaskFinal(id, ft)).catch(() => {});
        }
      }
    };
    sseES.onerror = () => { /* 连接断开时浏览器会自动重连; 已关闭的不再处理 */ };
  }
}
// 渲染任务终态: 结果摘要 + 错误
function renderTaskFinal(id, t) {
  const sum = document.getElementById('t-summary');
  if (sum) {
    // 有摘要就显示 (done 或 全部出错的 failed 都可能有摘要)
    sum.innerHTML = t.summary ? `<div class="card"><div class="card-title">结果摘要</div><div class="table-wrap"><table><thead><tr><th>评测集</th><th>阶段</th><th>分数</th><th>样本数</th></tr></thead><tbody>
      ${t.summary.map(s => `<tr><td class="cell-wrap">${esc(s.benchmark)}</td><td class="muted">${esc(s.stage)}</td><td><b>${esc(s.score)}</b></td><td>${s.num_samples}</td></tr>`).join('')}</tbody></table></div></div>` : '';
  }
  const err = document.getElementById('t-error');
  if (err) err.innerHTML = t.error ? `<div class="alert err"><b>错误:</b> ${esc(t.error)}</div>` : '';
}
// 拉取历史日志 (已完成任务用, 填充日志区便于查看报错)
async function loadTaskLogs(id) {
  try {
    const logs = await api.logs(id);
    logs.forEach(lg => appendLog(lg.level, lg.message, lg.ts));
  } catch (e) { /* 忽略 */ }
}
// 渲染各评测集进度条 (每个数据集一行: 名字[可点] + done/total + 百分比条 + 已完成集评分)
// 点数据集名字 → 跳该集每条请求明细页 (/requests, 与完整报告同一套渲染); 运行中/已结束都能点
// (已结束走完整 JSON 结果, 运行中走已完成样本实时追加)。
function renderBenchProgress(prog) {
  const el = document.getElementById('t-bench-prog');
  if (!el || !prog) return;
  _tdLastProg = prog;
  const entries = Object.entries(prog);
  if (!entries.length) { el.innerHTML = ''; return; }
  el.innerHTML = '<div style="font-size:14px;color:var(--muted);margin-bottom:8px">各评测集进度</div>' +
    entries.map(([b, p]) => {
      const pct = p.pct || 0;
      const stColor = p.status === 'done' ? 'var(--good)' : (p.status === 'running' ? 'var(--accent)' : (p.status === 'cancelled' ? 'var(--muted)' : 'var(--muted)'));
      const label = p.status === 'running' ? `${p.done}/${p.total}` : (p.status === 'done' ? `${p.total}条 ✓` : (p.status === 'cancelled' ? '已取消' : '等待中'));
      // 名字可点 (该集有样本: done 且 total>0, 或运行中 done>0); cancelled/等待中不可点
      const hasData = (p.status === 'done' && (p.total || 0) > 0) || ((p.done || 0) > 0);
      const nameHtml = hasData
        ? `<a class="bench-link" style="font-size:15px" onclick="viewBenchSamples('${esc(b)}')" title="查看 ${esc(b)} 每条请求明细"><b>${esc(b)}</b></a>`
        : `<span style="font-size:15px">${esc(b)}</span>`;
      // 已完成集: 分数 (按钮去掉, 点名字即可进明细)
      let extra = '';
      if (p.status === 'done' && p.score != null) {
        extra = `<span style="margin-left:10px">分数 <b style="font-size:15px;color:var(--good)">${esc(p.score)}</b></span>`;
      }
      return `<div style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;font-size:15px;margin-bottom:3px;align-items:center">
          <span>${nameHtml}${extra}</span><span style="color:${stColor}">${label} · ${pct}%</span>
        </div>
        <div class="bar" style="height:6px;margin:0"><div style="width:${pct}%;background:${stColor};height:6px;border-radius:3px"></div></div>
      </div>`;
    }).join('');
}
// 查看/切换某集的已完成样本明细 → 跳转独立明细页 (同报告样式)
function viewBenchSamples(bench) {
  location.hash = `#/tasks/${_tdTaskId}/samples/${encodeURIComponent(bench)}`;
}

// ----------------------------- 单集每条请求明细 (独立页, 与完整报告同一套渲染) -----------------------------
// 用 rpt_table.js 的 initRptTable 渲染 (报告 HTML 也用它), 格式100%统一。
// 数据来自 /api/tasks/{id}/requests: 运行中=已完成样本(ended=false), 已结束=完整结果(ended=true)。
let _sp = null;  // 明细页状态: { taskId, bench, ctrl, timer, ended }
async function renderSamplePage(id, bench) {
  if (sseES) { sseES.close(); sseES = null; }
  _spStopPoll();
  _sp = { taskId: id, bench, ctrl: null, timer: null, ended: false };
  setMain(`<h1>每条请求明细</h1>
    <div class="sub"><a href="#/tasks/${id}">← 返回任务详情</a> · <span id="sp-status" class="muted">加载中...</span></div>
    <div id="sp-host" class="card"><div class="hint">加载中...</div></div>`);
  await _spLoad();
  _spMaybePoll();
}
// 拉取请求明细数据 (报告同款 flat_reqs) 并用 initRptTable 渲染
async function _spLoad() {
  try {
    const data = await api.requests(_sp.taskId, _sp.bench);
    _spRender(data);
  } catch (e) {
    const host = document.getElementById('sp-host');
    if (host) host.innerHTML = `<div class="alert err">${esc(e.message)}</div>`;
  }
}
// 把 /requests 返回的数据渲染到表格 (首次+轮询共用; 控制器在首条数据到达时建一次, 之后用 refresh)
function _spRender(data) {
  const host = document.getElementById('sp-host');
  if (!host) return;
  _sp.ended = !!data.ended;
  const st = document.getElementById('sp-status');
  if (st) st.textContent = _sp.ended ? '完整结果 (任务已完成)' : '已完成样本 (任务运行中, 实时追加)';
  const flat = data.flat_reqs || [];
  const benches = data.bench_options || [];
  if (!flat.length) {
    // 无数据: 清掉旧表格, 显示空态 (并丢弃旧 ctrl/构建记录, 之后有数据时重建)
    _sp.ctrl = null; _sp.builtBenches = null; _sp.builtEnded = null;
    host.innerHTML = '<div class="empty">该集暂无已完成的样本</div>';
    return;
  }
  // 控制器只在"首次有数据"或"评测集集合/结束态变化"时重建 (否则下拉/分数卡不会更新)。
  // 其余轮询只 refresh 数据, 保留用户当前筛选/页码/展开。
  const needRebuild = !_sp.ctrl
    || _sp.builtBenches !== benches.length
    || _sp.builtEnded !== _sp.ended;
  if (needRebuild) {
    _sp.ctrl = initRptTable(host, flat, benches, data.bench_infos || {},
      { showToolbar: true, initialBench: _sp.bench, fetchDetail: _spFetchDetail });
    _sp.builtBenches = benches.length;
    _sp.builtEnded = _sp.ended;
  } else {
    _sp.ctrl.refresh(flat);
  }
}
// 点 Hash 展开时按需拉单条完整明细 (prompt/response/reasoning), 返回 Promise<完整row>
// rpt_table.js 的 fetchDetail 回调: (hash, liteRow) → Promise<fullRow>
function _spFetchDetail(hash, row) {
  const sid = row && row.sample_id;
  if (!sid) return Promise.resolve(null);
  return api.requestSample(_sp.taskId, sid).catch(() => null);
}
// 运行中任务: 轮询刷新 (拉新数据, initRptTable 会重渲染表格; 展开状态由其内部 expanded 维护)
async function _spRefresh() {
  if (_sp.ended) return;
  // 用户正在明细区交互时跳过本次刷新: rptRender 会重建 tbody, 清掉浏览器文本选区,
  // 导致复制输出/思维链时被 5s 轮询打断。检测两种"正在交互"信号:
  // 1) 有非折叠的文本选区 (用户正在拖选/已选中要复制的内容)
  // 2) 焦点落在明细区内 (在 pre 里光标定位、或输入框聚焦)
  // 满足任一则延后到下次轮询, 不丢数据 (下次 refresh 会带上累积的新样本)。
  if (_spUserInteracting()) return;
  try {
    const data = await api.requests(_sp.taskId, _sp.bench);
    if (data.ended) {
      // 任务刚结束: 停轮询, 用完整数据重渲染 (含分数); ctrl 仍复用, 只刷数据
      _spStopPoll();
    }
    _spRender(data);
  } catch (e) { /* 轮询失败静默, 下次重试 */ }
}
// 判断用户是否正与明细表格交互 (有选区或焦点在明细区), 用于决定是否跳过轮询刷新。
function _spUserInteracting() {
  const host = document.getElementById('sp-host');
  if (!host) return false;
  // 1) 非折叠的文本选区, 且选区落在明细区内 (正在选/已选要复制的内容)
  const sel = window.getSelection && window.getSelection();
  if (sel && !sel.isCollapsed && sel.rangeCount > 0) {
    let node = sel.anchorNode;
    while (node) {
      if (node === host) return true;
      node = node.parentNode;
    }
  }
  // 2) 焦点在明细区内 (在 pre 里光标定位、或输入框聚焦)
  const ae = document.activeElement;
  if (ae && host.contains(ae)) return true;
  return false;
}
function _spMaybePoll() {
  _spStopPoll();
  if (_sp.ended) return;
  // 仅运行中任务才轮询
  api.task(_sp.taskId).then(t => {
    if (t && (t.status === 'pending' || t.status === 'running') && !_sp.ended) {
      _sp.timer = setInterval(_spRefresh, 5000);
    }
  }).catch(() => {});
}
function _spStopPoll() { if (_sp && _sp.timer) { clearInterval(_sp.timer); _sp.timer = null; } }
function appendLog(level, msg, ts) {
  const el = document.getElementById('t-log');
  if (!el) return;
  const cls = level === 'error' ? 'err' : (level === 'warn' ? 'warn' : '');
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = (ts ? `[${ts.slice(11, 19)}] ` : '') + msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}
async function cancelTask(id) {
  if (!confirm('取消该任务?')) return;
  try { await api.cancelTask(id); } catch (e) { alert(e.message); }
}

// ----------------------------- 开发模式 -----------------------------
function renderDev() {
  setMain(`<h1>开发模式</h1><div class="sub">快速验证评测集 / 新特性 / 调试 bug</div>
    <div class="card">
      <div class="card-title">1. Dry-Run (不调真实 API)</div>
      <div class="hint">用模拟响应跑通评分+报告全流程, 验证新增评测集是否注册/评分正确。</div>
      <label>评测集 (逗号分隔)</label><input type="text" id="d_benches" placeholder="mmlu, gsm8k">
      <label>采样条数</label><input type="number" id="d_limit" value="5">
      <button class="btn" onclick="doDryRun()">执行 dry-run</button>
      <div id="d_result" style="margin-top:12px"></div>
    </div>
    <div class="card">
      <div class="card-title">2. 快速小样本试跑 (调真实 API, limit 小)</div>
      <div class="hint">填真实 API, 一键用小样本快速验证模型/评测集是否跑得通。</div>
      <div class="row">
        <div><label>模型名</label><input type="text" id="q_model" placeholder="glm-5.2-fp8"></div>
        <div><label>Base URL</label><input type="text" id="q_base" placeholder="https://..."></div>
      </div>
      <div class="row">
        <div><label>API Key</label><input type="password" id="q_key" placeholder="sk-..." autocomplete="off"></div>
        <div><label>采样条数</label><input type="number" id="q_limit" value="5"></div>
      </div>
      <label>评测集 (逗号分隔)</label><input type="text" id="q_benches" placeholder="mmlu, gsm8k">
      <button class="btn" onclick="doQuick()">开始试跑</button>
      <div id="q_result" style="margin-top:12px"></div>
    </div>
    ${S.user.role === 'admin' ? `<div class="card">
      <div class="card-title">3. 热加载评测集插件 (仅管理员)</div>
      <div class="hint">开发时新增/修改 benchmarks/ 下评测集后, 不重启即生效。</div>
      <button class="btn secondary" onclick="doReload()">热加载</button>
      <div id="r_result" style="margin-top:12px"></div>
    </div>
    <div class="card">
      <div class="card-title">4. 平台自检 (仅管理员)</div>
      <button class="btn secondary" onclick="doHealth()">自检</button>
      <div id="h_result" style="margin-top:12px"></div>
    </div>` : '<div class="alert">热加载与平台自检仅管理员可用。</div>'}`);
}
async function doDryRun() {
  const benches = document.getElementById('d_benches').value.split(',').map(s => s.trim()).filter(Boolean);
  const limit = parseInt(document.getElementById('d_limit').value) || 5;
  if (!benches.length) return alert('填评测集');
  document.getElementById('d_result').innerHTML = '<div class="hint">执行中...</div>';
  try {
    const r = await api.dryRun({ model_name: 'dry-run', benchmarks: benches, limit });
    document.getElementById('d_result').innerHTML = r.ok
      ? `<div class="table-wrap"><table><thead><tr><th>评测集</th><th>样本</th><th>分数</th></tr></thead><tbody>${r.brief.map(b => `<tr><td class="cell-wrap">${esc(b.benchmark)}</td><td>${b.num_samples}</td><td><b>${esc(b.score)}</b></td></tr>`).join('')}</tbody></table></div>`
      : `<div class="alert err">${esc(r.error)}</div>`;
  } catch (e) { document.getElementById('d_result').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
}
async function doQuick() {
  const model = document.getElementById('q_model').value.trim();
  const base = document.getElementById('q_base').value.trim();
  const key = document.getElementById('q_key').value;
  const benches = document.getElementById('q_benches').value.split(',').map(s => s.trim()).filter(Boolean);
  const limit = parseInt(document.getElementById('q_limit').value) || 5;
  if (!model || !base || !key || !benches.length) return alert('请填完整');
  document.getElementById('q_result').innerHTML = '<div class="hint">已提交, 跳转任务详情...</div>';
  try {
    const t = await api.quickSample({
      model_config: { name: model, base_url: base, api_key: key, model },
      benchmarks: benches, limit, streaming: false,
    });
    location.hash = '#/tasks/' + t.id;
  } catch (e) { document.getElementById('q_result').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
}
async function doReload() {
  document.getElementById('r_result').innerHTML = '<div class="hint">加载中...</div>';
  try {
    const r = await api.reloadBench();
    document.getElementById('r_result').innerHTML = `✓ 已加载 ${r.count} 个评测集${r.errors.length ? '<div class="alert err">' + r.errors.map(e => esc(e.module + ': ' + e.error)).join('<br>') + '</div>' : ''}`;
    S.benchmarks = r.benchmarks;
  } catch (e) { document.getElementById('r_result').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
}
async function doHealth() {
  document.getElementById('h_result').innerHTML = '<div class="hint">检查中...</div>';
  try {
    const h = await api.health();
    document.getElementById('h_result').innerHTML = `<div class="card" style="margin:0"><pre class="mono">${esc(JSON.stringify(h, null, 2))}</pre></div>`;
  } catch (e) { document.getElementById('h_result').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
}

// ----------------------------- 管理后台 -----------------------------
async function renderAdmin() {
  // 先加载用户列表 (供任务筛选下拉 + 显示用户名)
  const us = await api.adminUsers().catch(() => []);
  S._users = us;  // 缓存, 任务表显示用户名用
  const userOpts = ['<option value="">全部用户</option>'].concat(
    us.map(u => `<option value="${u.id}">${esc(u.username)}${u.role !== 'user' ? ' (' + u.role + ')' : ''}</option>`)
  ).join('');
  setMain(`<h1>管理后台</h1><div class="sub">用户 / 任务 / 平台监控</div>
    <div class="card"><div class="card-title">平台统计</div><div id="a_stats"><div class="hint">加载中...</div></div></div>
    <div class="card"><div class="card-title">用户管理</div><div id="a_users"><div class="hint">加载中...</div></div></div>
    <div class="card"><div class="card-title">全部任务 (所有用户)</div>
      <div class="req-toolbar" style="margin-bottom:12px">
        <input type="text" id="af_q" placeholder="🔍 搜索任务名 / 评测集..." oninput="adminFilterTasks()" style="flex:1;min-width:180px">
        <label>用户: <select id="af_user" onchange="adminFilterTasks()">${userOpts}</select></label>
        <label>状态: <select id="af_status" onchange="adminFilterTasks()">
          <option value="">全部</option><option value="done">完成</option><option value="failed">失败</option>
          <option value="running">运行中</option><option value="pending">排队</option><option value="cancelled">已取消</option>
        </select></label>
        <span id="af_count" class="muted" style="font-size:12px"></span>
      </div>
      <div id="a_tasks"><div class="hint">加载中...</div></div>
    </div>
    <div class="card"><div class="card-title">♻️ 回收站 (30天后彻底清除)</div><div id="a_trash"><div class="hint">加载中...</div></div></div>`);
  // 统计
  try {
    const s = await api.adminStats();
    document.getElementById('a_stats').innerHTML = `<div class="row">
      <div class="card" style="margin:0"><div class="muted">用户数</div><div style="font-size:24px;font-weight:700">${s.users}</div></div>
      <div class="card" style="margin:0"><div class="muted">任务总数</div><div style="font-size:24px;font-weight:700">${s.tasks_total}</div></div>
      <div class="card" style="margin:0"><div class="muted">运行中</div><div style="font-size:24px;font-weight:700">${s.running_now}</div></div>
      <div class="card" style="margin:0"><div class="muted">状态分布</div><div class="mono">${esc(JSON.stringify(s.tasks_by_status))}</div></div>
    </div>`;
  } catch (e) { document.getElementById('a_stats').innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
  // 用户
  renderAdminUsers(us);
  // 全部任务 (初始全量)
  await adminFilterTasks();
  // 回收站
  await loadTrash();
}
async function loadTrash() {
  const el = document.getElementById('a_trash');
  if (!el) return;
  try {
    const ts = await api.trash();
    const users = S._users || [];
    const uname = (id) => { const u = users.find(x => x.id === id); return u ? esc(u.username) : 'user_' + id; };
    el.innerHTML = ts.length ? `<div class="table-wrap"><table><thead><tr><th>ID</th><th>用户</th><th>名称</th><th>评测集</th><th>状态</th><th>删除时间</th><th>操作</th></tr></thead><tbody>${ts.map(t => `<tr>
      <td>${t.id}</td><td>${uname(t.user_id)}</td><td class="cell-wrap">${esc(t.name)}</td><td class="muted cell-bench">${esc((t.benchmarks||[]).join(', '))}</td><td class="cell-status">${badge(t.status)}</td>
      <td class="muted">${esc((t.deleted_at || '').slice(0, 19).replace('T', ' '))}</td>
      <td class="actions cell-actions">
        <button class="btn secondary" onclick="restoreTask(${t.id})">恢复</button>
        <button class="btn danger" onclick="hardDeleteTask(${t.id},'${esc(t.name)}')">彻底删除</button>
      </td></tr>`).join('')}</tbody></table></div>
      <div class="hint" style="margin-top:8px;display:flex;justify-content:space-between;align-items:center;gap:8px"><span>回收站任务 30 天后自动彻底清除 (含报告文件)。</span><button class="btn danger" onclick="emptyTrash()">全部清空</button></div>` : '<div class="empty">回收站为空</div>';
  } catch (e) { el.innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
}
async function restoreTask(id) {
  try { await api.restoreTask(id); loadTrash(); } catch (e) { alert('恢复失败: ' + e.message); }
}
async function hardDeleteTask(id, name) {
  if (!confirm(`彻底删除 "${name}"? 此操作不可恢复, 报告文件也会清除。`)) return;
  try { await api.hardDeleteTask(id); loadTrash(); } catch (e) { alert('删除失败: ' + e.message); }
}
async function emptyTrash() {
  if (!confirm('清空回收站? 将永久清除全部任务及其报告文件, 不可恢复。')) return;
  try { await api.emptyTrash(); loadTrash(); } catch (e) { alert('清空失败: ' + e.message); }
}
function renderAdminUsers(us) {
  const me = S.user || {};
  const myLvl = roleLvl(me.role);
  document.getElementById('a_users').innerHTML = `<div class="table-wrap"><table><thead><tr><th>ID</th><th>用户名</th><th>角色</th><th>状态</th><th>创建</th><th>操作</th></tr></thead><tbody>${us.map(u => {
    const tLvl = roleLvl(u.role);
    const canMgmt = myLvl > tLvl;            // 严格高于才能改密/禁用
    const canRole = me.role === 'super' && u.role !== 'super';  // 仅 super 且非 super 目标可升降级
    // 升降级按钮: admin<->user 互转
    let roleBtn = '';
    if (canRole) {
      const to = u.role === 'admin' ? 'user' : 'admin';
      roleBtn = `<button class="btn secondary" onclick="setRole(${u.id},'${to}')">${u.role === 'admin' ? '降为普通用户' : '升为管理员'}</button>`;
    }
    const mgmtBtns = canMgmt ? `
      <button class="btn secondary" onclick="setActive(${u.id},${!u.is_active})">${u.is_active ? '禁用' : '启用'}</button>
      <button class="btn secondary" onclick="resetPassword(${u.id},'${esc(u.username)}')">重置密码</button>` : '';
    const actions = (roleBtn || mgmtBtns) ? `${roleBtn}${mgmtBtns}` : '<span class="muted">-</span>';
    const roleBadge = u.role === 'super' ? 'failed' : (u.role === 'admin' ? 'running' : 'pending');
    return `<tr>
      <td>${u.id}</td><td>${esc(u.username)}${u.id === me.id ? ' <span class="muted">(你)</span>' : ''}</td>
      <td><span class="badge ${roleBadge}">${esc(ROLE_LABEL[u.role] || u.role)}</span></td>
      <td>${u.is_active ? '<span class="badge done">启用</span>' : '<span class="badge failed">禁用</span>'}</td>
      <td class="muted">${esc((u.created_at || '').slice(0, 19).replace('T', ' '))}</td>
      <td class="actions cell-actions">${actions}</td></tr>`;
  }).join('')}</tbody></table></div>`;
}
let _adminTaskCache = [];
let adminPage = 1, adminPageSize = 20;
function adminFilterParams() {
  const q = (document.getElementById('af_q').value || '').trim();
  const uid = document.getElementById('af_user').value;
  const st = document.getElementById('af_status').value;
  const params = {};
  if (q) params.q = q;
  if (uid) params.user_id = uid;
  if (st) params.status = st;
  return params;
}
async function adminFilterTasks() {
  // 筛选条件变化: 重置到第 1 页
  adminPage = 1;
  await loadAdminTasks();
}
async function loadAdminTasks() {
  const el = document.getElementById('a_tasks');
  if (!el) return;
  const params = adminFilterParams();
  const reqParams = { ...params, page: adminPage, page_size: adminPageSize };
  try {
    const data = await api.adminTasksPage(reqParams);
    _adminTaskCache = data.items;
    const users = S._users || [];
    const uname = (id) => { const u = users.find(x => x.id === id); return u ? esc(u.username) : 'user_' + id; };
    document.getElementById('af_count').textContent = `共 ${data.total} 条`;
    el.innerHTML = (data.items.length ? `<div class="table-wrap"><table><thead><tr><th>ID</th><th>用户</th><th>名称</th><th>评测集</th><th>状态</th><th>分数</th><th>创建</th><th>操作</th></tr></thead><tbody>${data.items.map(t => {
      const isActive = t.status === 'pending' || t.status === 'running';
      const actBtns = `<a class="btn secondary" href="#/tasks/${t.id}">查看</a>`
        + (isActive ? `<button class="btn secondary" onclick="adminStopTask(${t.id})">停止</button>` : '')
        + `<button class="btn danger" onclick="adminDelTask(${t.id},'${esc(t.name)}')">删除</button>`;
      return `<tr>
      <td>${t.id}</td><td>${uname(t.user_id)}</td><td class="cell-wrap">${esc(t.name)}</td><td class="muted cell-bench">${esc((t.benchmarks||[]).join(', '))}</td><td class="cell-status">${badge(t.status)}${t.status==='failed'&&t.error?`<br><span class="muted" style="font-size:11px" title="${esc(t.error)}">${esc(t.error.slice(0,30))}…</span>`:''}</td>
      <td>${t.status==='failed'?'<span class="badge bad">失败</span>':(avgScore(t.summary)!=null?`<b style="font-size:15px">${avgScore(t.summary)}</b><div class="muted" style="font-size:11px">${t.summary.length} 集平均</div>`:'-')}</td>
      <td class="muted">${esc((t.created_at || '').slice(0, 19).replace('T', ' '))}</td>
      <td class="actions cell-actions">${actBtns}</td></tr>`;
    }).join('')}</tbody></table></div>` : '<div class="empty">无匹配任务</div>')
      + renderPager(adminPage, data.total, adminPageSize, 'adminGoto', 'adminSetPageSize');
  } catch (e) { el.innerHTML = `<div class="alert err">${esc(e.message)}</div>`; }
}
async function adminGoto(p) { adminPage = Math.max(1, parseInt(p) || 1); await loadAdminTasks(); }
async function adminSetPageSize(ps) { adminPageSize = parseInt(ps); adminPage = 1; await loadAdminTasks(); }
async function setRole(id, role) { try { await api.setRole(id, role); const us = await api.adminUsers(); renderAdminUsers(us); S._users = us; } catch (e) { alert(e.message); } }
async function setActive(id, v) { try { await api.setActive(id, v); const us = await api.adminUsers(); renderAdminUsers(us); } catch (e) { alert(e.message); } }
async function resetPassword(id, username) {
  const np = prompt(`重置用户 "${username}" 的密码 (至少6位):`);
  if (!np) return;
  if (np.length < 6) return alert('密码至少 6 位');
  try { await api.resetPassword(id, np); alert(`已重置 ${username} 的密码`); }
  catch (e) { alert('重置失败: ' + e.message); }
}
// 管理后台: 停止/删除任意用户任务 (后端按管理岗放行)
async function adminStopTask(id) {
  if (!confirm(`停止任务 #${id}?`)) return;
  try { await api.cancelTask(id); adminFilterTasks(); }
  catch (e) { alert('停止失败: ' + e.message); }
}
async function adminDelTask(id, name) {
  if (!confirm(`删除任务 #${id} "${name}"? 将移入回收站, 30天后彻底清除。`)) return;
  try { await api.deleteTask(id); adminFilterTasks(); loadTrash(); }
  catch (e) { alert('删除失败: ' + e.message); }
}

// ----------------------------- 启动 -----------------------------
window.addEventListener('hashchange', route);
window.addEventListener('DOMContentLoaded', route);

// Ctrl/Cmd+A 范围收敛: 在明细 <pre> 或可编辑元素里按"全选"时, 只选该元素内容,
// 而非整个页面 (默认行为会选中文档全部, 复制输入/输出/思维链时很难受)。
// 判定目标元素优先级: ① 焦点元素是可编辑框 → 选它; ② 当前选区(鼠标点击定位)落在
// 某 <pre> 内 → 选该 <pre>。pre 默认不可聚焦, 故用选区锚点而非 activeElement 来定位。
window.addEventListener('keydown', function (e) {
  if ((e.ctrlKey || e.metaKey) && (e.key === 'a' || e.key === 'A')) {
    const ae = document.activeElement;
    // ① 可编辑元素: input/textarea/contenteditable, 全选其内容
    if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA' || ae.isContentEditable)) {
      const sel = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(ae);
      sel.removeAllRanges();
      sel.addRange(range);
      e.preventDefault();
      return;
    }
    // ② 当前选区锚点落在 <pre> 内: 全选该 <pre> 内容 (鼠标点进 pre 文本时, 浏览器
    //    会在 pre 内放一个折叠选区作光标定位, 锚点即指向该 pre)
    const sel = window.getSelection && window.getSelection();
    if (sel && sel.rangeCount > 0) {
      let node = sel.anchorNode;
      while (node && node.tagName !== 'PRE') node = node.parentNode;
      if (node && node.tagName === 'PRE') {
        const range = document.createRange();
        range.selectNodeContents(node);
        sel.removeAllRanges();
        sel.addRange(range);
        e.preventDefault();
      }
    }
  }
});
