"""统一 OpenAI 兼容 API 客户端

DeepSeek / Kimi(Moonshot) / Qwen(通义) / GLM(智谱) / OpenAI 等均提供 OpenAI 兼容的
chat/completions 接口, 这里用一个客户端统一覆盖, 不依赖任何厂商 SDK。

只用到标准库 + requests, 保证 Python 3.9 可用且依赖极轻。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

from .models import ModelConfig


class LLMClientError(Exception):
    pass


class LLMClient:
    """OpenAI 兼容的 chat completions 客户端

    用法:
        client = LLMClient(model_config)
        text, usage = client.chat("你好")
    """

    def __init__(self, config: ModelConfig, timeout: int = 1200, max_retries: int = 3, concurrency: int = 4):
        self.config = config
        # timeout 是 requests 的"两次数据之间"超时, 对慢吐型响应(思维链模型
        # 持续缓慢吐 token, 每次间隔 < timeout)无法兜底总耗时, 单题可挂几十分钟。
        # 故额外加 _hard_deadline: 单次请求(含重试)的总耗时硬上限, 超过强制中断。
        self.timeout = timeout
        self.max_retries = max_retries
        # 单次请求硬超时(秒): 兜底慢吐/挂起, 避免占满并发槽拖垮整个任务。
        # 同时作为"请求总耗时上限"——requests 的 read timeout 只管两次数据之间的
        # 间隔, 对持续慢吐的响应(每次间隔 < read timeout)无法兜底总耗时, 单题可挂
        # 几十分钟。用 _hard_timeout 作总预算, 超过即放弃(daemon 线程跑 post, 主线程
        # fut.result(timeout=) 强弃)。
        # 默认 1200s(20min, 与 config.yaml run.timeout 一致): 覆盖 AIME 思维链长题
        # (~600s)。用户可在提交表单按端点情况调大(端点 stall 时给更多恢复时间)或调小
        # (stall 期端点 0 字节, 超时快释放换下一题)。跟随 Runner 传入的 timeout。
        self._hard_timeout = timeout
        # 可选的取消信号 (threading.Event): 由 runner 注入 taskman 的 cancel_event。
        # 流式/非流式请求在 socket 读循环里检查它, 被 set 时立即关闭底层连接并抛
        # LLMClientError("已取消"), 让点"取消任务"能秒级中断在途请求, 而非等硬超时。
        self.cancel_event = None
        # 当前在途的 response 集合 (线程安全): 取消时由 abort_in_flight() 遍历 close(),
        # 强制中断卡在 iter_lines/recv 的 worker 线程。iter_lines 阻塞在 socket recv 时
        # 循环内的 cancel 检查进不去, 必须从外部关闭 socket 才能让它抛异常退出。
        self._active_resps = []  # list[requests.Response]
        self._resps_lock = __import__("threading").Lock()
        # 用独立 Session 并 trust_env=False: 显式绕过系统代理(http_proxy/https_proxy
        # 等环境变量)。被测端点通常是公网/内网直连地址, 走系统代理(Clash/Mihomo 等)
        # 会在大并发长连接评测时把连接"吊住"——本地 socket 既不可读也不报错(ESTABLISHED
        # 但收发队列空), 导致 socket timeout / fut.result / urllib3 read timeout 全部
        # 失效, worker 线程永久卡在 sock_recv_into。实测直连稳定, 代理是致因。
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.proxies = {"http": None, "https": None}
        # 连接池调优: requests 默认 urllib3 PoolManager 每 host maxsize=10, 而评测
        # 并发常达 32。并发 > 池大小时, urllib3 默认 pool_block=False 会为超出部分狂建
        # 临时连接(不进池, 用完即弃), 导致瞬时连接数飙升——在 BBH 等 GEN 类长思维链
        # 大集(6511 条 × 长连接)上, 持续打满端点连接数, 触发端点主动 RST/断连
        # (ConnectionResetError/RemoteDisconnected, 任务 #65 BBH 1328 条)。
        # 改: 池大小跟随用户并发数 (取 max(concurrency,32) 再 ×1.5 留余量), pool_block=True
        # 让超出请求排队复用 keep-alive 连接, 而非狂建临时连接。降低端点看到的并发连接数,
        # 缓解 RST。池跟随并发的好处: 用户设任意并发(如 100)连接池都不会成瓶颈——固定值
        # (如写死 64) 在并发>池时会变瓶颈(排队而非扩池)。judge_client 并发低, 用默认即可。
        _pool = max(int(concurrency * 1.5), 32)
        _adapter = HTTPAdapter(
            pool_connections=_pool,  # 连接池缓存数 (urllib3 连接数)
            pool_maxsize=_pool,      # 池最大连接数, ≥ 评测并发, 跟随用户设置
            pool_block=True,         # 池满时排队等待, 不新建临时连接
        )
        self._session.mount("http://", _adapter)
        self._session.mount("https://", _adapter)
        base = config.base_url.rstrip("/")
        # 兼容用户填 /v1 或不填: 若 URL 没有任何路径段则补 /v1
        # (DeepSeek/Kimi/Qwen 等官方文档均以 /v1 开头)
        path = base.split("//", 1)[-1].split("/", 1)[-1] if "//" in base else ""
        if not path:
            base = base + "/v1"
        self.endpoint = f"{base}/chat/completions"

    def _headers(self) -> dict:
        """构造请求头。api_key 为空时不发 Authorization (本地/自建无鉴权端点)。"""
        h = {"Content-Type": "application/json"}
        if self.config.api_key:
            h["Authorization"] = f"Bearer {self.config.api_key}"
        return h

    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """单轮对话。返回 (回答文本, usage字典)

        usage: {prompt_tokens, completion_tokens, total_tokens} (可能为 {})
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        # 合并模型级 extra (如 top_p 等)
        if self.config.extra:
            payload.update(self.config.extra)
        if extra:
            payload.update(extra)

        headers = self._headers()

        last_err: Optional[str] = None
        req_start = time.time()
        for attempt in range(1, self.max_retries + 1):
            # 取消检查: 用户点"取消任务"时 cancel_event 被 set, 立即中断, 不发新请求。
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise LLMClientError("已取消")
            # 剩余硬超时预算: 整个请求(含重试)的总耗时上限, 超过则放弃。
            # 思维链模型慢吐型响应会让 requests 的 timeout(两次数据间隔)永不触发,
            # 且 requests.post 可能在 recv 阶段对半开连接卡死不返回。
            # 用 daemon 线程跑 requests.post, 主线程 fut.result(timeout=) 超时后直接放弃——
            # daemon 线程不 join, 不会阻塞调用方, 进程退出时自动清理。
            remaining = self._hard_timeout - (time.time() - req_start)
            if remaining <= 0:
                raise LLMClientError(
                    f"请求硬超时({self._hard_timeout}s): {last_err or '慢吐/挂起'}"
                )
            # 单次 post 的 read timeout 取 min(剩余预算, self.timeout), 双保险。
            # 注意: 必须用 self._session(trust_env=False) 而非 requests.post, 否则会
            # 读取 http_proxy/https_proxy 环境变量走系统代理——代理会把连接"吊住"使
            # socket timeout 永不触发(见 __init__ 注释)。
            this_timeout = (10, min(remaining, self.timeout))  # (connect, read)
            t0 = time.time()
            resp = None
            # holder 供超时后强制关闭底层连接: worker 把已拿到的 response 放进来,
            # 主线程 fut.result 超时后调 close() 中断卡在 recv 的 worker。
            holder: Dict[str, Any] = {}

            def _do_post() -> Any:
                r = self._session.post(
                    self.endpoint, headers=headers, json=payload, timeout=this_timeout
                )
                holder["resp"] = r
                return r

            ex = ThreadPoolExecutor(max_workers=1)
            fut = ex.submit(_do_post)
            try:
                resp = fut.result(timeout=remaining + 5)
            except FutureTimeoutError:
                last_err = f"请求硬超时({self._hard_timeout}s, 慢吐/挂起)"
                # 强制关闭底层连接: 中断 worker 线程里卡在 sock_recv 的 requests.post。
                # holder["resp"] 通常为空(post 还没返回), 此时无法关闭, 只能放弃——
                # daemon 线程由系统在进程退出时回收。直连场景下 socket timeout 会先
                # 触发, 走不到这里; 此处仅为代理/半开连接等极端情况的兜底。
                try:
                    if "resp" in holder:
                        holder["resp"].close()
                except Exception:  # noqa: BLE001
                    pass
                ex.shutdown(wait=False)
                raise LLMClientError(last_err)
            except Exception as e:  # noqa: BLE001
                # _do_post 在 daemon 线程内抛出的网络异常 (ReadTimeout/ConnectionError
                # /ChunkedEncodingError 等) 会原样冒泡到这里。若不转成 LLMClientError,
                # 上层 worker 的 `except LLMClientError` 接不住 -> worker 抛出 ->
                # run_concurrent 把它 catch 成 {"_error":..} dict -> runner 过滤掉 ->
                # 样本蒸发、分数偏(分子分母都缩水)且报告里看不到失败样本。
                # 这里统一转成 LLMClientError, 让 worker 能正常构造错误 SampleResult。
                try:
                    if "resp" in holder:
                        holder["resp"].close()
                except Exception:  # noqa: BLE001
                    pass
                ex.shutdown(wait=False)
                raise LLMClientError(f"network: {e}") from e
            ex.shutdown(wait=False)
            latency = (time.time() - t0) * 1000
            try:
                if resp.status_code == 429 or resp.status_code >= 500:
                    # 限流/服务端错误 -> 指数退避重试
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    self._backoff(attempt)
                    continue
                if resp.status_code != 200:
                    raise LLMClientError(
                        f"HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                return self._parse_response(resp.json(), latency)
            except requests.RequestException as e:
                last_err = f"network: {e}"
                self._backoff(attempt)
                continue
        raise LLMClientError(f"请求失败(重试{self.max_retries}次后): {last_err}")

    @staticmethod
    def _parse_response(data: Dict[str, Any], latency: float) -> Tuple[str, Dict[str, int]]:
        """解析 OpenAI 兼容响应。

        - 取 message.content 作为最终答案 (思维链模型的思考在 reasoning_content,
          不计入答案)
        - finish_reason 一并放进 usage, 便于 runner 判断是否被 max_tokens 截断
        - completion_tokens_details.reasoning_tokens (若有) 记录思维链 token 消耗
        - reasoning_content (思维链全文) 也放进 usage.reasoning_content, 供记录/查看
        """
        choice = data["choices"][0]
        msg = choice.get("message", {}) or {}
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        usage = data.get("usage", {}) or {}
        out: Dict[str, Any] = {"latency_ms": latency}
        out.update({k: v for k, v in usage.items() if isinstance(v, (int, float))})
        out["finish_reason"] = choice.get("finish_reason")
        # 思维链 token 消耗 (vLLM/OpenAI 新版字段, 可能为空)
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        if isinstance(details, dict) and "reasoning_tokens" in details:
            out["reasoning_tokens"] = details["reasoning_tokens"]
        # 完整思维链文本 (思维链模型才有), 保存供请求浏览器查看
        if reasoning:
            out["has_reasoning"] = True
            out["reasoning_content"] = reasoning
        return text, out

    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """多轮对话版本 (MT-Bench 等多轮评测用)"""
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if self.config.extra:
            payload.update(self.config.extra)
        if extra:
            payload.update(extra)
        headers = self._headers()
        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._session.post(
                    self.endpoint, headers=headers, json=payload, timeout=self.timeout
                )
                latency = (time.time() - t0) * 1000
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    self._backoff(attempt)
                    continue
                if resp.status_code != 200:
                    raise LLMClientError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                return self._parse_response(resp.json(), latency)
            except requests.RequestException as e:
                last_err = f"network: {e}"
                self._backoff(attempt)
                continue
        raise LLMClientError(f"请求失败(重试{self.max_retries}次后): {last_err}")

    def chat_stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """流式对话。返回 (回答文本, usage字典)

        相比 chat(), 流式能拿到真实的:
        - ttft_ms: 首 chunk 到达时间 (真实首字延迟, 非近似)
        - gen_time_ms: 生成阶段耗时 (最后chunk - 首chunk)
        - tpot_ms: 每个输出 token 的生成时间 = gen_time / (completion_tokens - 1)
        思维链模型: 首 chunk 通常是 reasoning_content 的第一个片段, 故 ttft 仍为真实首字。
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if stop:
            payload["stop"] = stop
        if self.config.extra:
            payload.update(self.config.extra)
        if extra:
            payload.update(extra)
        headers = self._headers()

        last_err: Optional[str] = None
        req_start = time.time()
        for attempt in range(1, self.max_retries + 1):
            # 取消检查: 用户点"取消任务"时 cancel_event 被 set, 立即中断。
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise LLMClientError("已取消")
            # 剩余硬超时预算: 流式同样需要总耗时上限, 慢吐/挂起时强制中断释放并发槽。
            remaining = self._hard_timeout - (time.time() - req_start)
            if remaining <= 0:
                raise LLMClientError(
                    f"请求硬超时({self._hard_timeout}s): {last_err or '慢吐/挂起'}"
                )
            t0 = time.time()
            # holder 供超时/取消后强制关闭底层连接: worker 把 response 放进来,
            # 主线程 fut.result 超时或 cancel 后调 close() 中断卡在 iter_lines 的 worker。
            holder: Dict[str, Any] = {}
            cancel = self.cancel_event

            def _do_stream() -> Any:
                t_first: Optional[float] = None
                t_last: Optional[float] = None
                content_parts: List[str] = []
                reasoning_parts: List[str] = []
                finish_reason: Optional[str] = None
                usage: Dict[str, Any] = {}
                r = self._session.post(
                    self.endpoint, headers=headers, json=payload,
                    timeout=(10, min(remaining, self.timeout)), stream=True,
                )
                holder["resp"] = r  # 供主线程超时/取消时 close()
                with self._resps_lock:
                    self._active_resps.append(r)
                try:
                    if r.status_code == 429 or r.status_code >= 500:
                        return ("retry", f"HTTP {r.status_code}: {r.text[:200]}")
                    if r.status_code != 200:
                        return ("error", f"HTTP {r.status_code}: {r.text[:500]}")
                    # 逐行读取 SSE; 每行检查一次 cancel, 被 set 时立即返回取消标记
                    for line in r.iter_lines(decode_unicode=True):
                        if cancel is not None and cancel.is_set():
                            return ("cancelled", None)
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            now = time.time()
                            if t_first is None:
                                t_first = now
                            t_last = now
                            choices = chunk.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta", {}) or {}
                                if delta.get("content"):
                                    content_parts.append(delta["content"])
                                if delta.get("reasoning_content"):
                                    reasoning_parts.append(delta["reasoning_content"])
                                if choices[0].get("finish_reason"):
                                    finish_reason = choices[0]["finish_reason"]
                            if chunk.get("usage"):
                                usage = chunk["usage"]
                    latency = (time.time() - t0) * 1000
                    text = "".join(content_parts)
                    reasoning = "".join(reasoning_parts)
                    out: Dict[str, Any] = {"latency_ms": latency}
                    out.update({k: v for k, v in usage.items() if isinstance(v, (int, float))})
                    out["finish_reason"] = finish_reason
                    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
                    if isinstance(details, dict) and "reasoning_tokens" in details:
                        out["reasoning_tokens"] = details["reasoning_tokens"]
                    if reasoning:
                        out["has_reasoning"] = True
                        out["reasoning_content"] = reasoning
                    # 真实流式指标
                    # 注意: 部分服务(如某些 vLLM 部署)虽支持 stream=true, 但实际是"伪流式"——
                    # 先把整个响应算完, 再把 chunks 在极短时间内批量吐出。此时 gen_time(末chunk-首chunk)
                    # 极小, TPOT = gen_time/(tokens-1) 会小到不合理(如 0.01ms), 不能反映真实生成速度。
                    # 检测: 若 gen_time < 总耗时的 5%, 判定为伪流式, 不输出 TPOT/gen_time(置 None),
                    # 只保留有意义的 TTFT(首字延迟)。
                    if t_first is not None and t_last is not None:
                        out["ttft_ms"] = round((t_first - t0) * 1000, 2)
                        gen_time_ms = (t_last - t_first) * 1000
                        # 伪流式检测
                        is_fake_stream = gen_time_ms < latency * 0.05 and latency > 200
                        if not is_fake_stream:
                            out["gen_time_ms"] = round(gen_time_ms, 2)
                            comp_tok = usage.get("completion_tokens")
                            if comp_tok and comp_tok > 1:
                                out["tpot_ms"] = round(gen_time_ms / (comp_tok - 1), 2)
                        out["real_stream"] = not is_fake_stream  # 供报告判断是否显示 TPOT
                    return ("ok", (text, out))
                finally:
                    # 无论正常结束/取消/异常, 都从活跃集合移除并关闭连接。
                    with self._resps_lock:
                        if r in self._active_resps:
                            self._active_resps.remove(r)
                    try:
                        r.close()
                    except Exception:  # noqa: BLE001
                        pass

            ex = ThreadPoolExecutor(max_workers=1)
            fut = ex.submit(_do_stream)
            try:
                kind, payload2 = fut.result(timeout=remaining + 5)
            except FutureTimeoutError:
                last_err = f"请求硬超时({self._hard_timeout}s, 慢吐/挂起)"
                try:
                    if "resp" in holder:
                        holder["resp"].close()
                except Exception:  # noqa: BLE001
                    pass
                ex.shutdown(wait=False)
                raise LLMClientError(last_err)
            except Exception as e:  # noqa: BLE001
                # _do_stream 在 daemon 线程内 (iter_lines 消费 SSE 时) 抛出的网络异常
                # (ReadTimeout/ConnectionError/ChunkedEncodingError/ProtocolError 等)
                # 会原样冒泡到这里。不转成 LLMClientError 则上层 worker 接不住,
                # 样本会蒸发 (见 chat() 同位置注释)。流式尤其常见: 端点 stall 时
                # iter_lines 卡在 socket recv, read timeout 触发抛 ReadTimeout。
                try:
                    if "resp" in holder:
                        holder["resp"].close()
                except Exception:  # noqa: BLE001
                    pass
                ex.shutdown(wait=False)
                raise LLMClientError(f"network: {e}") from e
            ex.shutdown(wait=False)
            # 取消时也关闭连接 (worker 可能已返回 cancelled 标记, 或主线程检测到)
            if cancel is not None and cancel.is_set():
                try:
                    if "resp" in holder:
                        holder["resp"].close()
                except Exception:  # noqa: BLE001
                    pass
                raise LLMClientError("已取消")
            if kind == "cancelled":
                raise LLMClientError("已取消")
            if kind == "retry":
                last_err = payload2
                self._backoff(attempt)
                continue
            if kind == "error":
                raise LLMClientError(payload2)
            return payload2  # ("ok", (text, out))
        raise LLMClientError(f"请求失败(重试{self.max_retries}次后): {last_err}")

    @staticmethod
    def _backoff(attempt: int) -> None:
        # 指数退避: 1s, 2s, 4s ...
        time.sleep(min(2 ** (attempt - 1), 16))

    def abort_in_flight(self) -> None:
        """取消时强制中断所有在途请求的底层 socket。

        iter_lines 阻塞在 urllib3 的 read_chunked → socket.recv, requests 的
        r.close() / os.close(fd) 都无法中断它 (urllib3 缓存了 socket 引用, 且
        close 不唤醒阻塞的 recv)。唯一可靠的方法是拿到底层 socket 调
        shutdown(SHUT_RDWR) —— 这会让阻塞的 recv 立即返回, iter_lines 抛异常退出。
        由 runner 在 cancel_event 触发后调用 (run_concurrent 的 should_stop 分支)。
        """
        import socket as _socket
        with self._resps_lock:
            resps = list(self._active_resps)
            self._active_resps.clear()
        # 1) 已拿到 response (在 iter_lines 读 body) 的: shutdown 其底层 socket
        for r in resps:
            sock = None
            # urllib3 HTTPResponse → _fp(BufferedReader) → raw → _sock; 多层 try 兼容版本差异
            try:
                if r.raw is not None:
                    sock = r.raw._fp.fp.raw._sock  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            if sock is not None:
                try:
                    sock.shutdown(_socket.SHUT_RDWR)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    sock.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                if r.raw is not None:
                    r.close()
            except Exception:  # noqa: BLE001
                pass
        # 2) 还卡在 post (等响应头/建连) 阶段、没注册 resp 的 worker: 它们的 socket
        # 在 session 的连接池里, 逐个取不到。直接 close 整个 session 的所有连接——
        # 会中断池中所有 in-use socket 的阻塞 recv, 让 post 抛异常退出。
        # (取消时整个 client 要停, 不需要复用连接, 关掉无副作用。)
        try:
            self._session.close()
        except Exception:  # noqa: BLE001
            pass


def run_concurrent(
    items: List[Any],
    worker: callable,
    concurrency: int = 4,
    on_progress: Optional[callable] = None,
    should_stop: Optional[callable] = None,
    on_cancel: Optional[callable] = None,
) -> List[Any]:
    """并发执行 worker(item) -> result, 保持输入顺序返回结果。

    worker 抛出的异常会被捕获并以 {"_error": str} 形式返回对应位置。
    on_progress(done, total, idx, result): 每完成一条样本调用一次 (用于进度回调)。
    should_stop(): 返回 True 时, 取消尚未开始的任务并立即返回——不等 in-flight 完成。
    on_cancel(): should_stop 首次触发时调用一次, 供调用方强制关闭 in-flight 的底层
    连接 (如 LLMClient.abort_in_flight), 让卡在 socket recv 的 worker 抛异常退出。
    """
    results: List[Any] = [None] * len(items)
    total = len(items)
    done = 0
    pool = ThreadPoolExecutor(max_workers=concurrency)
    future_to_idx = {pool.submit(worker, item): i for i, item in enumerate(items)}
    pending = set(future_to_idx)
    stopped = False
    grace_deadline = None  # 取消后给 in-flight 的宽限截止时间
    try:
        # 用 wait(timeout=) 轮询而非 as_completed 阻塞: 这样即使所有 in-flight 都卡住
        # (慢吐/挂起), 主线程也能每 0.5s 检查一次 should_stop, 取消时立即响应,
        # 而非等某个 future 完成才有机会检查。
        import time as _time
        while pending:
            if not stopped and should_stop and should_stop():
                stopped = True
                # 强制中断 in-flight 请求的底层连接: iter_lines/post 阻塞在 socket 时,
                # worker 内的 cancel 检查进不去, 必须从外部 shutdown socket 让它抛异常。
                if on_cancel is not None:
                    try:
                        on_cancel()
                    except Exception:  # noqa: BLE001
                        pass
                # 取消尚未开始的 future; 已 in-flight 的由 abort + worker 内 cancel_event
                # 协同中断。给 2 秒宽限期让 in-flight worker 因 socket 中断而完成。
                for f, i in future_to_idx.items():
                    if results[i] is None and not f.done():
                        f.cancel()
                        if f.cancelled():
                            results[i] = {"_error": "已取消", "_cancelled": True}
                grace_deadline = _time.time() + 2.0
            # 取消宽限期过后仍有 pending: 不再等 (worker 卡在 abort 无法中断的极端情况,
            # 如建连阶段), 直接交给 shutdown(wait=False), 主流程立即返回。
            if stopped and grace_deadline and _time.time() > grace_deadline:
                break
            done_set, pending = wait(pending, timeout=0.5)
            for fut in done_set:
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    results[idx] = {"_error": str(e)}
                done += 1
                if on_progress:
                    try:
                        on_progress(done, total, idx, results[idx])
                    except Exception:  # noqa: BLE001
                        pass  # 回调失败不影响主流程
    finally:
        # shutdown(wait=False, cancel_futures=True): 不阻塞等待 in-flight 线程。
        # 已取消的 future 立即释放; 仍 in-flight 的 worker 线程是 daemon 性质, 不 join
        # 主流程, 会在 socket 中断后或进程退出时自行结束。这样"取消任务"能秒级返回。
        pool.shutdown(wait=False, cancel_futures=True)
    # 取消后未完成的样本标记为已取消 (worker 未及响应的 in-flight)
    if stopped:
        for i, r in enumerate(results):
            if r is None:
                results[i] = {"_error": "已取消", "_cancelled": True}
    return results
