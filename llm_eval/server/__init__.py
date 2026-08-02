"""服务端平台 (生产用)

把单机单用户的评测工具升级成多用户在线平台:
- db.py     SQLite 持久层 (用户/任务/会话/日志)
- auth.py   注册/登录/会话/角色 + pbkdf2 密码哈希
- schemas.py Pydantic 请求/响应模型
- taskman.py 任务管理器 (队列 + worker 线程 + SSE 事件总线)
- routes/   路由分组 (auth/task/dev/admin)
- app.py    FastAPI 工厂: 挂载路由 + 静态 SPA + 启动时 admin 引导

被测模型 API Key 只在任务运行期间驻留内存, 绝不落库 (见 auth/taskman)。
"""
