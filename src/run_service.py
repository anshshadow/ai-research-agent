"""
Script to run the FastAPI service.
"""
import uvicorn

from core import settings


if __name__ == "__main__":
    uvicorn.run(
        "service.service:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
