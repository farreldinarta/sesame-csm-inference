import io
import inspect
from fastapi.responses import StreamingResponse
from typing import Any
from tests.utils.exceptions.test_exception import TestFailureException
from app.provider.llm.interface import LLMInterface

class SesameCSMLLMProviderMock(LLMInterface): 
    def __init__(self):
      self.__inference_returning_value : Any = None
      self.__inference_expected_arg : Any = None

    def set_inference_func_behaviour(self, expected_arg : str, returning : bool):
      self.__inference_expected_arg = expected_arg
      self.__inference_returning_value = returning
      return self

    async def inference(self, prompt : str) -> io.BytesIO:
      if (prompt != self.__inference_expected_arg):
        raise TestFailureException(f"""
          {inspect.currentframe().f_code.co_name} function argument mismatch :
          # Expected argument : '{self.__inference_expected_arg}' 
          # Got : '{prompt}'
        """) 

      return self.__inference_returning_value