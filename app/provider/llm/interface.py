from abc import ABC, abstractmethod

class LLMInterface(ABC):

  @abstractmethod
  async def inference(self, prompt : str) -> bool:
    pass