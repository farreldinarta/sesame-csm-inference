from fastapi import Depends
from app.utils.object import singleton
from app.provider.llm.interface import LLMInterface
from app.provider.llm.sesame_csm_llm_provider import get_sesame_csm_model

@singleton
class LLM:
  def __init__(self, sesame_csm: LLMInterface):
    self.__sesame_csm = sesame_csm

  def get_sesame_csm_model(self) -> LLMInterface:
    return self.__sesame_csm
  

def get_llm(sesame_csm : LLMInterface = Depends(get_sesame_csm_model)) -> LLM:
  return LLM(sesame_csm=sesame_csm)