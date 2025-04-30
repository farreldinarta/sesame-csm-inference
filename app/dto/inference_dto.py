from pydantic import BaseModel

class GenerateCodeInferenceRequest(BaseModel):
  text : str

class GenerateCodeInferenceResponse(BaseModel):
  data : str