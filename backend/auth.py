import json
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

from logger import get_logs

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 密钥和算法
SECRET_KEY = "greeneye"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
USER_DB = "users.json"


# 加载用户数据
def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=2)


def authenticate_user(username: str, password: str):
    users = load_users()
    if username in users and pwd_context.verify(password, users[username]["password"]):
        return {"username": username}
    return None


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload.get("sub")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# 登录接口
# @router.post("/token")
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = authenticate_user(form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(401, detail="Incorrect username or password")
#     token = create_access_token({"sub": user["username"]})
#     return {"access_token": token, "token_type": "bearer"}


# 注册接口
# @router.post("/register")
# async def register(username: str, password: str):
#     users = load_users()
#     if username in users:
#         raise HTTPException(400, detail="Username already exists.")
#     hashed = pwd_context.hash(password)
#     users[username] = {"password": hashed}
#     save_users(users)
#     return {"status": "ok", "message": "User registered."}


@router.get("/admin/logs/")
def fetch_logs(limit: int = 100):
    return JSONResponse(content={"logs": get_logs(limit)})


from typing import Optional

from fastapi import Body, Path

# 获取所有用户（仅用户名）
# @router.get("/admin/users/")
# def list_users():
#     users = load_users()
#     return {"users": list(users.keys())}


# 创建新用户（等同于注册）
# @router.post("/admin/users/")
# def create_user(username: str = Body(...), password: str = Body(...)):
#     users = load_users()
#     if username in users:
#         raise HTTPException(400, detail="User already exists.")
#     hashed = pwd_context.hash(password)
#     users[username] = {"password": hashed}
#     save_users(users)
#     return {"status": "ok", "message": f"User '{username}' created."}


# 更新用户密码
# @router.put("/admin/users/{username}")
# def update_user(username: str = Path(...), new_password: str = Body(...)):
#     users = load_users()
#     if username not in users:
#         raise HTTPException(404, detail="User not found.")
#     hashed = pwd_context.hash(new_password)
#     users[username]["password"] = hashed
#     save_users(users)
#     return {"status": "ok", "message": f"Password for '{username}' updated."}


# 删除用户
# @router.delete("/admin/users/{username}")
# def delete_user(username: str = Path(...)):
#     users = load_users()
#     if username not in users:
#         raise HTTPException(404, detail="User not found.")
#     del users[username]
#     save_users(users)
#     return {"status": "ok", "message": f"User '{username}' deleted."}
