from datasets import load_dataset
from datasets import ClassLabel, Sequence

raw_datasets = load_dataset("SameerMahajan/marathi_numbers-1-20")

def preprocess(batch):
    batch["number"] = batch["labels"]
    batch["labels"] =  [batch["labels"][0] - 1]
    return batch

raw_datasets = raw_datasets.map(preprocess)

# get labels from the dataset
label_names = sorted(set(label for labels in raw_datasets["train"]["labels"] for label in labels))

# Cast to ClassLabel
raw_datasets = raw_datasets.cast_column("labels", Sequence(ClassLabel(names=label_names)))

# push to hub
raw_datasets.push_to_hub("SameerMahajan/marathi_numbers-1-20")