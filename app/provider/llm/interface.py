import io
from fastapi.responses import StreamingResponse
from abc import ABC, abstractmethod

class LLMInterface(ABC):

  @abstractmethod
  async def inference(self, prompt : str) -> io.BytesIO:
    pass