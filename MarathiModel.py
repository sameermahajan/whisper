import torch
from transformers import pipeline

model_id = "shripadbhat/whisper-tiny-mr"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
   "automatic-speech-recognition",
   model=model_id,
   device=device,
)

for number in range(1,21,1):
    for i in range(50):
        audio_file = './samples/' + str(number) + '/' + str(number) + '_' + str(i) + '.wav'
        if os.path.isfile(audio_file):
            out = pipe(audio_file)
            print (audio_file, out)
