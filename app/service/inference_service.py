from fastapi import Depends
from app.dto.base import APIResponse
from app.dto.inference_dto import GenerateCodeInferenceRequest, GenerateCodeInferenceResponse
from app.provider.llm.base import LLM, get_llm
from app.utils.object import singleton

@singleton
class InferenceService:
  def __init__(self, llm: LLM):
    self.__llm = llm

  async def generate(self, payload : GenerateCodeInferenceRequest) -> APIResponse[GenerateCodeInferenceResponse, str]:
    print("EHHEHE")
    test_bool : bool = await self.__llm.get_sesame_csm_model().inference(payload.text)
    if not test_bool:
      return APIResponse(
        message="Failed to generate code",
        error="Failed to generate code"
      )

    return APIResponse(
        message="Successfully generated code",
        data=GenerateCodeInferenceResponse(data="ABC123")
    )

def get_inference_service(llm : LLM = Depends(get_llm)) -> InferenceService:
  return InferenceService(llm=llm)