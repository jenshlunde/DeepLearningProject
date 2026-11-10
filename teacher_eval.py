import argparse
import json
from pathlib import Path

import numpy as np

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
OUT = Path("/home/dl08e26/OUTPUT")
DATA = Path("/home/dl08e26/DATA")
LEADERBOARD_AUC = 0.98856


def best_threshold(p, y):
    from sklearn.metrics import precision_recall_curve
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def report(P_val, Y_val, P_test, Y_test):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    rows = []
    print(f"{'label':14s} {'prev val/test':>14s} {'AUC val':>8s} {'AUC test':>9s} {'PR-AUC t':>9s} "
          f"{'F1@.5 val':>10s} {'F1@.5 test':>11s} {'thr(val)':>9s} {'F1 test@thr':>12s}")
    for j, name in enumerate(LABELS):
        yv, yt, pv, pt = Y_val[:, j], Y_test[:, j], P_val[:, j], P_test[:, j]
        thr = best_threshold(pv, yv)
        r = dict(
            label=name, prev_val=yv.mean(), prev_test=yt.mean(),
            auc_val=roc_auc_score(yv, pv) if 0 < yv.sum() < len(yv) else np.nan,
            auc_test=roc_auc_score(yt, pt) if 0 < yt.sum() < len(yt) else np.nan,
            pr_test=average_precision_score(yt, pt) if yt.sum() else np.nan,
            f1_val=f1_score(yv, pv >= 0.5, zero_division=0),
            f1_test=f1_score(yt, pt >= 0.5, zero_division=0),
            thr=thr, f1_test_tuned=f1_score(yt, pt >= thr, zero_division=0))
        rows.append(r)
        print(f"{name:14s} {r['prev_val']*100:6.2f}/{r['prev_test']*100:5.2f}% {r['auc_val']:8.4f} {r['auc_test']:9.4f} "
              f"{r['pr_test']:9.3f} {r['f1_val']:10.3f} {r['f1_test']:11.3f} {thr:9.2f} {r['f1_test_tuned']:12.3f}")
    m = lambda k: float(np.nanmean([r[k] for r in rows]))
    print(f"\nMEAN AUC   val {m('auc_val'):.4f}   test {m('auc_test'):.4f}   (leaderboard winner: {LEADERBOARD_AUC})")
    print(f"MACRO F1 @0.5  val {m('f1_val'):.3f}   test {m('f1_test'):.3f}   |   test with val-tuned thresholds {m('f1_test_tuned'):.3f}")
    return rows


def predict(model, tok, texts, device, batch_size, max_length):
    import torch
    order = np.argsort([len(t) for t in texts])             # length-sorted batches: much less padding
    probs = np.zeros((len(texts), 6), dtype=np.float32)
    model.eval().to(device)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            probs[idx] = torch.sigmoid(model(**enc).logits).float().cpu().numpy()
            if (s // batch_size) % 200 == 0:
                print(f"  {s}/{len(texts)}", flush=True)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", default=str(OUT / "teacher_TL"))
    ap.add_argument("--tag", default="teacher")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse", action="store_true", help="skip inference if saved predictions exist")
    a = ap.parse_args()

    import torch
    from torch.utils.data import random_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import jsonDataset

    dest = OUT / "diagnostics"
    dest.mkdir(exist_ok=True)
    f_val, f_test = dest / f"{a.tag}_val.npy", dest / f"{a.tag}_test.npy"

    # identical split to project_pipeline.py (90/10 random_split with the same seed)
    train_ds = jsonDataset.jigsawDataset(DATA / "jigsaw_data_train.jsonl")
    n_train = int(0.9 * len(train_ds))
    _, val_subset = random_split(train_ds, [n_train, len(train_ds) - n_train],
                                 generator=torch.Generator().manual_seed(a.seed))
    val_texts = [train_ds[i][0] for i in val_subset.indices]
    Y_val = np.array([train_ds.labels[i] for i in val_subset.indices])
    test_ds = jsonDataset.jigsawDataset(DATA / "jigsaw_data_test.jsonl")
    test_texts = [test_ds[i][0] for i in range(len(test_ds))]
    Y_test = np.array(test_ds.labels)
    print(f"val {len(val_texts)} samples, test {len(test_texts)} samples")

    if not (a.reuse and f_val.exists() and f_test.exists()):
        tdir = Path(a.teacher_dir)
        tok_dir = tdir / "TOKENIZER"
        tok = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else "answerdotai/ModernBERT-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            tdir / "MODEL", num_labels=6, problem_type="multi_label_classification")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("predicting validation split..."); np.save(f_val, predict(model, tok, val_texts, device, a.batch_size, a.max_length))
        print("predicting test set...");        np.save(f_test, predict(model, tok, test_texts, device, a.batch_size, a.max_length))
    rows = report(np.load(f_val), Y_val, np.load(f_test), Y_test)
    json.dump(rows, open(dest / f"{a.tag}_report.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
