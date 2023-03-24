from datasets import Dataset, Audio, concatenate_datasets

fnames = []
for i in range (50):
     fnames.append("./samples/" + str(1) + "/" + str(1) + "_" + str(i) + ".wav")
fnames.append("./samples/" + str(1) + "/" + str(1) + "_50.wav")
audio_dataset = Dataset.from_dict({"audio": fnames }).cast_column("audio", Audio())
audio_dataset = audio_dataset.add_column("labels", [[1]] * len(audio_dataset))

for number in range(2,21,1):
    fnames = []
    for i in range (50):
        fnames.append("./samples/" + str(number) + "/" + str(number) + "_" + str(number) + ".wav")
    fnames.append("./samples/" + str(number) + "/" + str(number) + "_50.wav")
    ds = Dataset.from_dict({"audio": fnames }).cast_column("audio", Audio())
    ds = ds.add_column("labels", [[number]] * len(ds))
    audio_dataset = concatenate_datasets([audio_dataset, ds])
audio_dataset.push_to_hub("SameerMahajan/marathi_numbers-1-20")
