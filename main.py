import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Suhhhhhhhhhhhhh"
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

AUDIO_PROMPT_PATH="sample.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)

AUDIO_PROMPT_PATH="dad_sample.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-3.wav", wav, model.sr)
