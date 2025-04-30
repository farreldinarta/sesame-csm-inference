import io
import torch
import torchaudio
from fastapi.responses import StreamingResponse
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

  async def inference(self, prompt : str) -> io.BytesIO:

    audio = self.__model.generate(
        prompt,
        speaker=0,
        context=[],
        max_audio_length_ms=90_000,
        temperature=0.9,
        topk=50
    )
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio.unsqueeze(0).cpu(), self.__model.sample_rate, format="wav")
    buffer.seek(0)  

    # Debug print: buffer size
    buffer.seek(0, io.SEEK_END)
    size = buffer.tell()
    print(f"[DEBUG] Buffer size: {size} bytes")

    # Debug print: preview first 20 bytes
    buffer.seek(0)
    preview = buffer.read(20)
    print(f"[DEBUG] First 20 bytes (raw): {preview}")

    # Optional: Base64 preview (more readable, but limited length)
    buffer.seek(0)
    base64_preview = base64.b64encode(buffer.read(60)).decode('utf-8')
    print(f"[DEBUG] First 60 bytes (base64): {base64_preview}")

    buffer.seek(0)  # Reset before returning    

    return buffer
  
def get_sesame_csm_model() -> LLMInterface:
  return SesameCSMLLMProvider()
