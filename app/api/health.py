from fastapi import APIRouter
from app.services import prediction_service

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Checks the health status of the API and its loaded components.
    """
    status = prediction_service.get_health_status()
    return status