from fastapi import APIRouter, Depends
from app.service.inference_service import InferenceService, get_inference_service
from app.dto.base import APIResponse
from app.dto.inference_dto import GenerateCodeInferenceRequest, GenerateCodeInferenceResponse

router = APIRouter()

@router.post('', response_model=APIResponse[GenerateCodeInferenceResponse, str])
async def generate(payload : GenerateCodeInferenceRequest, service : InferenceService = Depends(get_inference_service)) -> APIResponse[GenerateCodeInferenceResponse, str]:
  return await service.generate(payload)