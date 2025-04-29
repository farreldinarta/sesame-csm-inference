from pydantic import BaseModel

class GenerateCodeInferenceRequest(BaseModel):
  data : str

class GenerateCodeInferenceResponse(BaseModel):
  data : str