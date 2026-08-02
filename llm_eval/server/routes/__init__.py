"""路由分组

- auth_routes   注册/登录/登出/me
- task_routes   任务提交/列表/详情/删除/克隆/日志/报告/进度SSE
- dev_routes    开发模式 (dry-run/快速样本/热加载)
- admin_routes  管理员 (用户/任务/统计)

每个模块导出一个 register(app) 函数, 由 app.py 工厂挂载。
"""
