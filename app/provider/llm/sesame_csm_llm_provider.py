import torch
from app.utils.object import singleton
from app.provider.llm.interface import LLMInterface
from app.provider.llm.sesame_csm_utils.sesame_csm_generator import Generator, load_csm_1b
from app.configs.environment import get_environment_variables

env = get_environment_variables()
@singleton
class SesameCSMLLMProvider(LLMInterface):
  __device : str

  def __init__(self):

    if torch.backends.mps.is_available():
        self.__device = "mps"
    elif torch.cuda.is_available():
        self.__device = "cuda"
    else:
        self.__device = "cpu"

    print("Used Device : ", self.__device)
    
    self.__model = load_csm_1b(device=self.__device)

  async def inference(self, prompt : str) -> bool:
    self.__model = load_csm_1b(device=self.__device)
    return prompt == 'test'
  
def get_sesame_csm_model() -> LLMInterface:
  return SesameCSMLLMProvider()
