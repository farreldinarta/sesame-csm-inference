import io
from fastapi import Depends
from app.dto.inference_dto import GenerateCodeInferenceRequest
from app.provider.llm.base import LLM, get_llm
from app.exception.http_exception import HTTPException
from app.utils.object import singleton

@singleton
class InferenceService:
  def __init__(self, llm: LLM):
    self.__llm = llm

  async def generate(self, payload : GenerateCodeInferenceRequest) -> io.BytesIO:
    audio_buffer : io.BytesIO = await self.__llm.get_sesame_csm_model().inference(payload.text)
    if not audio_buffer:
      raise HTTPException(
        status=500,
        message="Failed to generate code",
        error=None
      )

    print(audio_buffer)
    return audio_buffer

def get_inference_service(llm : LLM = Depends(get_llm)) -> InferenceService:
  return InferenceService(llm=llm)