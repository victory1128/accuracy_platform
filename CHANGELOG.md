# 更新日志 (CHANGELOG)

本项目所有显著变更均记录于此。版本号遵循 [语义化版本 (SemVer)](https://semver.org/lang/zh-CN/),
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

---

## 版本管理规范

### 版本号 `MAJOR.MINOR.PATCH`

| 段 | 何时升 | 举例 |
|---|---|---|
| **MAJOR**(大版本) | 有**不兼容的破坏性变更**:API/配置/数据格式不向后兼容,或整体架构重大调整、里程碑式大改 | `1.x.x → 2.0.0` |
| **MINOR**(小版本) | **向后兼容的新功能**:新增评测集、新增端点、新增配置项、显著特性增强 | `1.0.x → 1.1.0` |
| **PATCH**(补丁) | **向后兼容的缺陷修复**:修 bug、修评分误判、性能优化、文档完善,无新功能 | `1.0.0 → 1.0.1` |

### 升版时机

- **每完成一批改动**(修复若干 bug 或上线一个新功能)→ 发版,更新本文件 + `llm_eval/__init__.py` 的 `__version__`,提交并打 git tag。
- **大的内容更新**(新增评测集、新的大特性、架构调整)→ 升 MINOR 或 MAJOR。
- **定期发版**:即便无大改动,每隔一段时间(如每月)累积的小修复也可发一个 PATCH/MINOR。
- **紧急修复**:线上 bug 修复后立即发 PATCH。

### 变更类型

- `新增` — 新功能 / 新评测集 / 新端点
- `变更` — 对已有功能的修改(向后兼容)
- `修复` — Bug 修复
- `优化` — 性能 / 体验提升,无行为变化
- `文档` — 文档变更
- `已弃用` / `移除` — 即将移除 / 已移除的功能

### 发版流程

```bash
# 1. 改 llm_eval/__init__.py 的 __version__
# 2. 在本文件顶部新增版本条目 (倒序, 最新在上)
# 3. 提交 + 打 tag + 推送
git add -A
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

---

## [1.0.6] — 2026-08-04

### 优化

- **连接池大小改为跟随用户并发数(v1.0.5 写死 64 的改进)**:v1.0.5 把连接池写死 `pool_maxsize=64`,虽对默认并发 32 够用,但用户若设并发 > 64(并发数是用户可配项),池会变瓶颈——`pool_block=True` 下超出请求只能排队而非扩池。改为:`LLMClient.__init__` 加 `concurrency` 参数,池大小 = `max(int(concurrency * 1.5), 32)`(跟随用户设置并留余量),`Runner` 构造主 client 时传入 concurrency(judge_client 并发低用默认)。用户设任意并发(8/64/100)连接池都自适应,不再有"并发 > 池大小"的瓶颈。

### 修复

- **DS-1000 沙箱 torch/tensorflow 安装方式落地(v1.0.5 增装方案实战调优)**:v1.0.5 在 Dockerfile 增装 torch/tensorflow 的初版方案(直接 pip install)经实战发现两处不可行,已修正:① torch 用 PyTorch 官方 CPU 源境外下载其纯 Python 依赖(networkx/fsspec/filelock/jinja2/MarkupSafe)常 ReadTimeout——改为本地预下载 `torch-2.13.0+cpu-...aarch64.whl` 到 `sandbox_wheels/` 经 COPY 安装(绕过 pip 下载不稳定),纯 Python 依赖走阿里云源;② `tensorflow-cpu` 变体只发布 linux_x86_64 wheel,本机构建架构是 aarch64(Apple Silicon)无任何匹配(pip 报 No matching distribution),改用主包 `tensorflow`(aarch64 有 wheel,无 GPU 环境天然 CPU-only)。注意 wheel 文件名须保留原始规范名(含版本/cp311/平台标签),pip 靠文件名解析元数据,改名会被拒。镜像体积 1.5GB→5.75GB。已验证:`--network none --read-only --mount tmpfs /tmp` 真实沙箱约束下 torch 2.13.0+cpu / tensorflow 2.21.0 / xgboost 3.2.0 / pyyaml 6.0.3 全部 import 且执行成功。`sandbox_wheels/`(161MB)不入 git(.gitignore 排除),重建镜像前需自行下载 wheel 到该目录(见 Dockerfile 注释)。

### 验证

- **BBH 连接池修复实跑验证(任务 #66)**:v1.0.5 的连接池修复 + 本版跟随并发数改进,经任务 #66 实跑验证——同端点 glm-5.2-fp8、同并发 32、streaming、全量 BBH 6511 条,**0 网络错误 / 0 空响应 / 0 失败**(对比 #65 修复前的 1328 条 RST/断连/超时),全程 1h17min,分数 **93.7%**(与 #65 排除网络故障后的真实 93.1% 一致,印证模型真实能力 ~93%,之前 74.1% 是网络错误拖累的假低分)。32 并发长连接全程无 RST,"并发 > 池 → 狂建临时连接刺激端点"的放大故障彻底消除。

## [1.0.5] — 2026-08-04

### 修复

- **请求连接池过小致端点连接重置放大(BBH 1328 条 RST)**:`client.py` 的 `requests.Session` 未挂载自定义 `HTTPAdapter`,用 urllib3 默认 `PoolManager`(每 host `maxsize=10`)。评测并发常达 32,并发 > 池大小时 urllib3 默认 `pool_block=False` 为超出部分狂建临时连接(不进池,用完即弃),瞬时连接数飙升。任务 #65 BBH 集(6511 条 GEN 类长思维链,长连接)上,持续打满 GLM 端点连接数,触发端点主动 RST/断连——`ConnectionResetError`(1076)/`RemoteDisconnected`(219)/`IncompleteRead`(1),共 1328 条拉低报告分至 74.1%(排除网络故障后真实 93.1%)。改:`LLMClient.__init__` 加 `concurrency` 参数,挂 `HTTPAdapter(pool_connections=max(并发×1.5,32), pool_maxsize=同, pool_block=True)`——池大小**跟随用户并发数**(非写死,用户设任意并发 8/64/100 连接池都自适应不成瓶颈),池满时排队复用 keep-alive 连接而非狂建临时连接,降低端点看到的并发连接数。`Runner` 构造主 client 时传入 concurrency(judge_client 并发低用默认)。流式请求中途断开仍不重试(已吐 chunk 不幂等,正确设计)。v1.0.1 的 `except Exception` 转 `LLMClientError` 已生效——1328 条错误样本全部保留入分母(非蒸发)。**验证(任务 #66)**:同端点同并发 32 重跑 BBH 全量 6511 条,**0 网络错误/0 空响应/0 失败**,全程 1h17min,分数 **93.7%**(与 #65 排除网络后的真实 93.1% 一致,印证模型真实能力 ~93%,放大故障已彻底消除)。
- **DS-1000 沙箱缺 torch/tensorflow(DS-1000 PyTorch/Tensorflow 子集环境误伤)**:`Dockerfile.code-sandbox` 原只预装 pandas/numpy/scipy/sklearn/matplotlib/seaborn 等,但 DS-1000 数据集本身含 PyTorch(68 题)/Tensorflow(45 题)子集,题目官方参考实现即用这两个框架,模型按题目语境 `import torch/tensorflow` 是正确的。沙箱没装 → `ModuleNotFoundError` 误判 fail(任务 #65 DS-1000 113 条环境误伤)。改:Dockerfile 增装 `torch`/`tensorflow`/`xgboost`/`pyyaml`(CPU 版,沙箱 `--network none` 禁网无 GPU)。安装方式经实战调优:① torch 用 PyTorch 官方 CPU 源境外下载其纯 Python 依赖(networkx/fsspec/filelock/jinja2/MarkupSafe)常 ReadTimeout——改为本地预下载 `torch-2.13.0+cpu-...aarch64.whl` 到 `sandbox_wheels/` 经 COPY 安装(绕过 pip 下载不稳定),纯 Python 依赖走阿里云源;② `tensorflow-cpu` 变体只发布 linux_x86_64 wheel,本机构建架构是 aarch64(Apple Silicon)无任何匹配(pip 报 No matching distribution),改用主包 `tensorflow`(aarch64 有 wheel,无 GPU 环境天然 CPU-only)。注意 wheel 文件名须保留原始规范名(含版本/cp311/平台标签),pip 靠文件名解析元数据,改名会被拒。镜像体积 1.5GB→5.75GB。已验证:`--network none --read-only --mount tmpfs /tmp` 真实沙箱约束下 torch 2.13.0+cpu / tensorflow 2.21.0 / xgboost 3.2.0 / pyyaml 6.0.3 全部 import 且执行成功。

## [1.0.4] — 2026-08-03

### 修复

- **LLM 裁判未适配思维链模型(自裁判全部失败)**:`scoring/judging.py` 的 `judge_single`/`judge_pair` 用 `max_tokens=256/128` 且丢弃 `usage`(只取 `text, _`),思维链裁判模型(如 glm-5.2-fp8 自裁判)会先在 `reasoning_content` 思考再在 `content` 输出分数/判定——256 token 预算被推理吃光致 `finish_reason=length`、content 为空,`_parse_judge_json("")` 返回 None,MT-Bench/AlpacaEval/Arena-Hard 全部判"裁判解析失败"(1635 样本空跑)。改为:`max_tokens` 256/128→4096 让思考走完;接收 `usage`,content 空/无 JSON 时从 `usage.reasoning_content` 末尾兜底抽取(JSON/score;配对比较找"因此选 A"或末尾独立 A/B 字母)。对照 `grade_simpleqa`(已是 max_tokens=4096 + reasoning 兜底)的正确范式补齐。真实端点验证:glm-5.2 自裁判正确出分(score=10.0)与判定(verdict=A)。非思维链裁判(content 正常)行为不变。

## [1.0.3] — 2026-08-03

### 修复

- **SimpleQA acceptable range 数字容差未检查**:部分数字题 gold 带 `(acceptable range: anything between X and Y)` 容差区间(88/1000 条),`_clean_gold` 删注释只比主答案,把落在区间内的 `33.7833`(范围 [33.7586, 33.8022])判错。新增 `_in_acceptable_range`:从原始 gold 取区间,检查 pred 首个数值是否落在 [X, Y] 内(去千分位逗号后匹配),区间外不救回。任务 #59 重算救回 5 条(45.7% → 46.8%)。
- **`normalize_answer` 不处理 Unicode 破折号/重音字母**:`string.punctuation` 只含 ASCII,en-dash `–`(U+2013)/em-dash/curly quote 不被清理,重音字母 `é`/`ä`/`ș` 不折叠。gold `Urbana–Champaign`(en-dash) vs pred `Urbana-Champaign`(hyphen)、`Viștea Mare` vs `Vistea Mare` 被误判不等。改为:NFKD 分解折叠重音(café→cafe)+ Unicode 标点映射表归一(en-dash/em-dash/curly quote → ASCII)。影响 SimpleQA 及所有英文评测集。任务 #59 救回 2 条。

## [1.0.2] — 2026-08-03

### 新增

- **单题超时时间用户可配置**:`client.py` 的请求硬超时原写死 `600s`(10 分钟),覆盖了 `config.yaml` 里 `run.timeout: 1200` 的配置——配置项实际是死参数,用户设 20 分钟也不生效。改为:硬超时跟随 `timeout` 参数,默认 `1200s`(20 分钟,与配置一致)。提交表单"运行参数"区新增「单题超时(秒)」输入框,留空=默认 1200;端点易 stall 时可调大(给恢复时间)或调小(stall 期端点 0 字节,超时快释放换下一题)。全链路贯通:`TaskCreateIn.timeout` → `run_params.timeout` → `taskman` 取 `rp.get('timeout') or 1200` → `Runner(timeout=)` → `LLMClient._hard_timeout`。clone 任务也能取回该值。

### 修复

- **硬超时与配置不一致(死参数)**:`client.py:42` `self._hard_timeout = 600` 写死,使 `config.yaml` 的 `timeout: 1200` 形同虚设——任务 #59 simpleqa 重跑日志显示 `read timeout=599.99s`(10 分钟)而非预期的 20 分钟,即因此。现 `_hard_timeout = timeout`,与配置/用户输入一致。

## [1.0.1] — 2026-08-03

### 新增

- **SimpleQA 评分改用 LLM 裁判三分类(官方做法)**:旧实现用归一化精确匹配,比官方 LLM judge 严苛,把 `"120,000"` vs `"120,000 euros"`、`"Oct 23"` vs `"October 23"` 这类语义等价但表述不同的正确答案判错,实测压低约 10 个百分点(37.6% → 真实约 47.2%)。改为:有裁判模型时用 LLM 做 A/B/C 三分类(正确/错误/未尝试,改编自 OpenAI simple-evals),三分类结果存 `analysis.simpleqa_grade` 供报告展示;无裁判时回退到宽松匹配(包含关系 + 数字归一)。思维链裁判模型(如 glm-5.2 自裁判)需 `max_tokens=4096` 让 reasoning 走完才在 content 输出字母——太小会被 length 截断致 content 空,故加 reasoning_content 末尾兜底抽取。
- **DS-1000 沙箱兼容旧版 pandas/numpy API**:DS-1000 题目写于 pandas<2.0/numpy<2.0 时代,沙箱是新版库,`DataFrame.append`/`np.NAN`/`np.in1d`/`read_csv(delim_whitespace=)`/`replace(method=)` 等被移除的 API 会让模型按题目语境写的正确代码报错(环境误伤,非模型能力问题,实测 23 条)。在执行 preamble 注入兼容 shim 复活这些 API,不改变新 API 行为。

### 修复

- **GSM8K 百分号/LaTeX 答案误判**:思维链模型输出常带 LaTeX 转义(`60\%`/`\92`)或百分号(`60%`),旧 `numbers_equal` 因含反斜杠判非纯数值直接字符串比较致不等,且把 `60%` 换算成 0.6 与 gold `60` 比致不等。新增 `_math_answer_equal` 兜底:清理 `\` 后重比;gold 为纯数字时去 `%` 比较(60%==60)。实测 95.8% → 97.6%(救回 23 条)。真错的(51 vs 91)仍判错。
- **DROP 单位/修饰词答案误判(重灾区)**:DROP 答案常带单位或修饰词(`24-yard`/`9 yards`/`Dallas Cowboys`)而 gold 是纯 span(`24`/`9`/`Dallas`),旧 `_drop_match` 只做精确匹配+numbers_equal 把这些判错,实测 **75.8% → 89.2%**(救回 1243 条,占错误样本一半以上)。改为:数字 gold 剥离 pred 单位后取首个数字组数值比较;纯文本 gold(len≥4)做包含关系。关键防误判:数字 gold(如 `2`,占 DROP 一半)绝不走文本包含,否则 `2` in `20 yards` 会误判——已用 `g_is_number` 分支隔离验证。

### 修复

- **超时/网络异常样本被静默丢弃致分数偏(分子分母缩水)**:任务 #58 全量重跑暴露——ds1000 报 33.3(453 条)、simpleqa 37.6(663 条),但全量应 1000 条,547/337 条超时样本彻底蒸发,分数是"幸存者偏分"且报告里看不到失败样本。根因两层:① `client.py` 的 `chat`/`chat_stream` 用 daemon 线程跑 `requests.post`/`iter_lines`,`fut.result()` 只 `except FutureTimeoutError`,而 daemon 线程内的网络异常(`ReadTimeout`/`ConnectionError`/`ChunkedEncodingError`)会原样冒泡未被转成 `LLMClientError`;② `runner.py` 用 `hasattr(r,"analysis")` 过滤,把这些异常样本(被 `run_concurrent` catch 成 `{"_error":..}` dict)整个丢弃。修复:client 层 `fut.result()` 后加 `except Exception` 把网络异常转 `LLMClientError`(chat + chat_stream 两处);runner 层过滤改为兜底——`zip(samples,results)` 配对把失败 dict 转成错误占位 `SampleResult`(correct=False,保留 sample_id/prompt/gold/error),计入分母、报告可追溯。两层防御:client 修根因,runner 兜底防 worker 漏出任何异常。此 bug 独立于端点问题,只要有一题超时就会发生,历史凡有超时的全量集分数均虚高。

## [1.0.0] — 2026-08-02

首个纳入 git 管理的正式版本。基于真实全量评测(13 万+样本,31 个评测集)打磨稳定,
覆盖预训练 + 后训练主流评测集,含自动评分、乱码分析、实时进度、多用户平台。

### 新增

- **平台核心**:FastAPI 多用户在线平台(注册/登录/HttpOnly 会话),REST API + 原生 JS SPA 前端,无构建步骤。
- **18+ 评测集**:预训练(MMLU/MMLU-Pro/GPQA/C-Eval/CMMLU/ARC/HellaSwag/WinoGrande/TruthfulQA/GSM8K/MATH-500/AIME/HumanEval/MBPP 等)+ 后训练(IFEval/MT-Bench/AlpacaEval/Arena-Hard),最终注册 34 个。
- **统一 API 客户端**:一套 OpenAI 兼容客户端覆盖 DeepSeek/Kimi/Qwen/GLM/OpenAI 等,不依赖厂商 SDK。
- **五种评分方式**:选项精确匹配 / 数字抽取匹配 / 代码执行 pass@1 / LLM 裁判打分 / 规则可验证。
- **乱码分析**:检测 mojibake 编码乱码、重复退化、控制字符、异常字符集、语言不一致、截断、空输出,给出健康度等级。
- **双入口**:命令行 CLI 自动化批跑 + Web 控制台可视化。
- **报告**:JSON 完整结果 + HTML 仪表板(每条请求明细、分页、搜索、展开看 prompt/思维链/输出)。
- **代码沙箱**:docker(禁网/只读根/资源上限,最强隔离)/ subprocess 双模式,执行模型生成的 Python 代码。
- **实时进度**:SSE 推送任务日志 + 数据集进度;单条样本跑完即可查看吐字明细,一集跑完即出评分(不必等全任务结束)。
- **增量落盘**:每跑完一个评测集即保存 JSON/HTML,中途异常不丢已完成结果。
- **任务管理**:提交/查看/克隆/删除/取消,回收站 30 天清理,仪表盘与管理后台分页(修 >200 截断)。
- **部署文档**:`DEPLOY.md` 新 Linux 服务器从零部署指南。

### 修复

- **MCQ 中文答案抽取漏判**:`经过分析，选A`/`故选C`/`应选 D` 抽不到字母导致全判错(影响 ceval/cmmlu/agieval)。加 `(?:故|应|因此|所以|则)?选` 模式 + valid_choices 过滤防误匹配。
- **思维链吃光 token 预算**:reasoning 过长致 content 空,旧逻辑把 reasoning 当 response 评分致开放 QA 全错。各评测集 max_tokens 统一提到 8192(AIME 竞赛题需 65536),保留 reasoning 回退兜底(MCQ/数学答案常在 reasoning 末尾)。
- **代码执行 NameError(两处)**:① MBPP test_code 用 `math.isclose` 却没 import → `run_code_tests` 加 `_ensure_stdlib_imports` 自动补标准库 import;② HumanEval prompt 里辅助函数(如 encode_shift)执行时丢失 → `_eval_code` 把含 `def` 的 prompt 作前缀拼到 completion 前。
- **乱码检测误报**:LaTeX 数学定界符 `$`、数学符号 `× ⁻ ·` 被归 "异常脚本" → `_script_of` 把 S 类符号归 "symbol";LaTeX `\text{ mile}` 反复触发 "高度重复" → ngram 30%-50% 降级为弱提示,≥50% 才报重复退化。
- **runner 单题慢吐卡死全局**:思维链模型持续慢吐 token 使 `requests.post(timeout)` 永不触发(该 timeout 是两次数据间间隔非总耗时),慢题占满并发槽致整个任务卡死。加请求级硬超时(`ThreadPoolExecutor` 包 post,`fut.result(timeout=remaining)` 超时强弃)。
- **系统代理吊死连接**:本机全局代理(http_proxy)致评测请求绕道代理,代理对大并发长连接"吊住"(ESTABLISHED 但收发队列全空,timeout 触发极慢)。`requests.Session(trust_env=False, proxies={http:None,https:None})` 强制直连。
- **取消任务不即时**:点取消后任务仍显示 running 几十分钟(等在途慢吐请求自然结束)。三层修复:流式 `iter_lines` 阻塞时用底层 socket `shutdown(SHUT_RDWR)` 中断(唯一可靠方式,`r.close()`/`os.close(fd)` 都无效);`run_concurrent` 改 `wait(timeout=0.5)` 轮询取消标志不等 in-flight;taskman 加 cancelled 终态判定。
- **livecodebench 驼峰函数名误判**:442 题中 440 个 entry_point 是驼峰(findPeaks/sumOfSquares),模型常写成蛇形(find_peaks),harness 按驼峰调用 → NameError 误判 fail。prompt 明示函数名 + 执行层 `_entry_point_alias` 自动补别名(驼峰↔蛇形归一化匹配,唯一候选才补)。
- **corpusqa 数据 context 全空**:HF 源语料在 `prompt` 消息列表的 user content 里,下载脚本误用 `context` 字段取不到 → 改遍历 `prompt` 取 role=user 的 content;配套改 corpusqa.py 支持多正确答案(任一匹配即对)。
- **报告分页跳转滚到网页顶**:`<tbody>` 的 `offsetTop` 在多数浏览器返回 0,改用 `#view-bench` 的 `getBoundingClientRect` 定位明细区开头。

### 优化

- **大任务明细页加载**:`/requests` 端点 lite 模式(不返回大文本,62MB→8.6MB)+ 点 Hash 按需拉单条完整明细 + 结果 JSON 缓存(按 mtime 失效)+ GZip 中间件(8.6MB→0.9MB)。万条级样本首屏不再卡死。
- **并发吞吐**:压测确定 32 并发为甜点(再高无收益),全量 13 万样本从 ~12 天压到 ~1 天。

### 已知限制

- 无断点续跑机制:任务中断后只能新建任务跑剩余集(增量落盘保已完成集不丢)。
- 思维链模型单题耗时长(AIME 最长 ~5 万 token),高并发长时间打满端点易触发 HTTP 429 限流,**一次只提交一个任务批次**。
- 超长上下文评测集(mrcr/corpusqa 单条达 132 万 token)超过模型输入上限会 ContextWindowExceeded,按模型能力判错(预期行为,反映真实能力边界)。
