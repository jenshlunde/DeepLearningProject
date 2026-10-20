"""Save per-sample test-set probabilities for the teacher and all 18 students.
Run from the project folder:  CUDA_VISIBLE_DEVICES=1 python predict_students.py
Output: OUTPUT/predictions/<model>.npy  (N x 6 sigmoid probabilities) + labels.npy
Row i is always test sample i (the test set is not shuffled)."""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import jsonDataset
import modelSetups

OUT = Path("/home/dl08e26/OUTPUT")
TEST = Path("/home/dl08e26/DATA/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_test.jsonl")
PRED = OUT / "predictions"
PRED.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = jsonDataset.jigsawDataset(TEST)


@torch.no_grad()
def predict(model, collator, batch_size, workers=4):
    model.eval().to(device)
    out = []
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, collate_fn=collator):
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(torch.sigmoid(model(**batch).logits).float().cpu())
    model.cpu()
    return torch.cat(out).numpy()


np.save(PRED / "labels.npy", np.array(ds.labels))

# Teacher (batch 8: the teacher reads uncapped, long comments)
tt = AutoTokenizer.from_pretrained(OUT / "teacher_TL" / "TOKENIZER")
teacher = AutoModelForSequenceClassification.from_pretrained(
    OUT / "teacher_TL" / "MODEL", num_labels=6, problem_type="multi_label_classification")
np.save(PRED / "teacher.npy", predict(teacher, jsonDataset.SupervisedCollator(tt), batch_size=8))
print("saved teacher", flush=True)
del teacher

# Students (batch 256 is fine: this is not a latency measurement)
st = AutoTokenizer.from_pretrained(OUT / "tokenizer" / "STUDENT")
info = json.load(open(OUT / "student_names.json"))
for name, path in zip(info["names"], info["paths"]):
    model = modelSetups.create_student_model(name, vocab_size=len(st))
    model.load_state_dict(load_file(str(Path(path) / "model.safetensors"), device="cpu"))
    np.save(PRED / f"{name}.npy", predict(model, jsonDataset.SupervisedCollator(st), batch_size=256))
    print("saved", name, flush=True)
