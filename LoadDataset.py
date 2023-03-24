from datasets import load_dataset

audio_dataset = load_dataset("SameerMahajan/marathi_numbers-1-20")
print (audio_dataset["labels"])