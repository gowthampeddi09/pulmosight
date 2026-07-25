import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import get_settings
from app.utils.logging import setup_logging
from app.api.v1.router import router as api_v1_router
from app.inference.model_loader import load_model

settings = get_settings()
setup_logging("DEBUG" if settings.debug else "INFO")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s application", settings.app_name)
    # Pre-warm model in background
    try:
        load_model()
        log.info("PyTorch model initialized during startup")
    except Exception as e:
        log.error("Failed to load model during startup: %s", e)
    yield
    log.info("Shutting down %s application", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.model_version,
    description="Advanced AI Medical Intelligence Platform — Chest X-Ray Pneumonia Detection",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production setup would filter origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Ensure validation errors conform to consistent JSON error shape."""
    log.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": {"code": "VALIDATION_ERROR", "message": str(exc.errors())}},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all unhandled exception handler."""
    log.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected server error occurred"}},
    )


# Register API Router
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
