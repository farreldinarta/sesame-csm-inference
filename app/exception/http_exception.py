from typing import Any
from fastapi import HTTPException as FastAPIHTTPException
from app.dto.base import APIResponse

class HTTPException(FastAPIHTTPException):
  def __init__(self, status : int, message : str, error : Any):
      self.api_response = APIResponse(
          message=message,
          data=None,
          error=error
      )
      super().__init__(status_code=status, detail=self.api_response.dict())