import torch
from transformers import pipeline

model_id = "DrishtiSharma/whisper-large-v2-marathi"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
   "automatic-speech-recognition",
   model=model_id,
   device=device,
)

for i in range(50):
    audio = './samples/2/2_' + str(i) + '.wav'
    out = pipe(audio)
    print (audio, out)
