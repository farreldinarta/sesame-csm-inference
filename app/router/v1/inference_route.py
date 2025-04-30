import io
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.service.inference_service import InferenceService, get_inference_service
from app.dto.inference_dto import GenerateCodeInferenceRequest, GenerateCodeInferenceResponse

router = APIRouter()

@router.post('', response_class=StreamingResponse)
async def generate(payload : GenerateCodeInferenceRequest, service : InferenceService = Depends(get_inference_service)) -> StreamingResponse:
  audio : io.BytesIO = await service.generate(payload)
  return StreamingResponse(audio, media_type="audio/wav")