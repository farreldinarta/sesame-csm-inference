import io
import os
import base64
import torch
import torchaudio
import tempfile
from app.utils.object import singleton
from app.provider.llm.interface import LLMInterface
from app.provider.llm.sesame_csm_utils.sesame_csm_generator import load_csm_1b, Segment
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

    self.__transcripts = [
       "Hello, how are you today? I hope you are doing well.",
       "Thank you for your cooperation, have a great day ahead!"
    ]

    self.__speakers = [0 for _ in self.__transcripts]

    audio_path = os.path.join("/code/app", "storage", "audio")
    self.__audio_paths = [
        os.path.join(audio_path, "ginny_sample_1.wav"),
        os.path.join(audio_path, "ginny_sample_2.wav")
    ]
    print(audio_path)

    self.__segments = [
        Segment(text=transcript, speaker=speaker, audio=self.__load_audio(audio_path))
        for transcript, speaker, audio_path in zip(self.__transcripts, self.__speakers, self.__audio_paths)
    ]
    
  def __load_audio(self, audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=self.__model.sample_rate
    )
    return audio_tensor    

  async def inference(self, prompt : str) -> io.BytesIO:

    audio = self.__model.generate(
        prompt,
        speaker=0,
        context=self.__segments,
        max_audio_length_ms=90_000,
        temperature=0.9,
        topk=50
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name

    torchaudio.save(temp_path, audio.unsqueeze(0).cpu(), self.__model.sample_rate)

    with open(temp_path, "rb") as f:
        wav_data = f.read()

    os.remove(temp_path)  

    buffer = io.BytesIO(wav_data)
    buffer.seek(0)

    print(f"[DEBUG] Buffer size: {len(wav_data)} bytes")
    print(f"[DEBUG] First 20 bytes (raw): {wav_data[:20]}")
    print(f"[DEBUG] First 60 bytes (base64): {base64.b64encode(wav_data[:60]).decode()}")

    return buffer
  
def get_sesame_csm_model() -> LLMInterface:
  return SesameCSMLLMProvider()
