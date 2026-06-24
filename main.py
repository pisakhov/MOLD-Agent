from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from app.database import init_db
from app.routes_agent import router as agent_router
from app.routes_models import router as models_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(agent_router, prefix="/api", tags=["agent"])
app.include_router(models_router, prefix="/api/models", tags=["models"])
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/settings")
def settings(request: Request):
    return templates.TemplateResponse(request, "settings.html")


@app.get("/global.css")
def global_css():
    return FileResponse("frontend/global.css")
