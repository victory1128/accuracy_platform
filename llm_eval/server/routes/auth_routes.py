"""认证路由: 注册 / 登录 / 登出 / 当前用户

会话 token 通过 HttpOnly Cookie 下发 (浏览器自动带), 同时支持
Authorization: Bearer <token> (SPA fetch / 非浏览器客户端)。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from .. import auth, db
from ..schemas import LoginIn, Message, PasswordChangeIn, RegisterIn, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserOut)
def register(body: RegisterIn):
    """注册。第一个用户自动成为 admin。"""
    u = auth.register(body.username, body.password)
    return u


@router.post("/login", response_model=UserOut)
def login(body: LoginIn, response: Response):
    """登录, 下发 HttpOnly session cookie (用 max_age 控制有效期)。"""
    user, token, expires_at = auth.login(body.username, body.password)
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=auth.SESSION_TTL_DAYS * 86400,
        path="/",
    )
    return user


@router.post("/logout", response_model=Message)
def logout(request: Request, response: Response):
    """登出, 删除会话 + 清 cookie。"""
    token = request.cookies.get("session")
    if token:
        auth.logout(token)
    response.delete_cookie("session", path="/")
    return Message(message="已登出")


@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(auth.current_user)):
    """当前登录用户。"""
    return auth._public_user(user)


@router.post("/password", response_model=Message)
def change_password(body: PasswordChangeIn, user: dict = Depends(auth.current_user)):
    """用户自己改密码: 需原密码 + 新密码 (前端输两次新密码校验一致)。"""
    auth.change_password(user, body.old_password, body.new_password)
    return Message(message="密码已修改")
