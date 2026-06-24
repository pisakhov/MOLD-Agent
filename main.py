from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from app.database import init_db
from app.routes_agent import router as agent_router
from app.routes_models import router as models_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    from app.store import seed_codex_from_auth
    seed_codex_from_auth()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    from app.auth import authed
    path = request.url.path
    if path == "/login" or path == "/global.css" or path.startswith("/static/") or authed(request):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return RedirectResponse("/login", status_code=303)

app.include_router(agent_router, prefix="/api", tags=["agent"])
app.include_router(models_router, prefix="/api/models", tags=["models"])
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend")


@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
def login_submit(request: Request, password: str = Form(...)):
    from app.auth import COOKIE_NAME, password as configured_password, token
    if password != configured_password():
        return templates.TemplateResponse(request, "login.html", {"error": "Incorrect password"}, status_code=401)
    response = RedirectResponse("/", status_code=303)
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(COOKIE_NAME, token(), httponly=True, secure=secure, samesite="lax", max_age=60 * 60 * 24 * 30)
    return response


@app.get("/logout")
def logout():
    from app.auth import COOKIE_NAME
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE_NAME)
    return response


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/settings")
def settings(request: Request):
    return templates.TemplateResponse(request, "settings.html")


@app.get("/global.css")
def global_css():
    return FileResponse("frontend/global.css")
