from fastapi import APIRouter
from app.router.v1.inference_route import router as inference_routes

router = APIRouter()

router.include_router(inference_routes, prefix='/inference')