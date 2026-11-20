"""Evaluate teachers + students of one or more pipeline runs ("variants") with ranking metrics, then plot and compare.

A VARIANT is one OUTPUT-style folder, laid out like the pipeline's output:
    <root>/teacher_TL/MODEL  (+ teacher_TL/TOKENIZER)
    <root>/best_students/<STUDENT_NAME>/model.safetensors
    <root>/student_names.json            (optional; otherwise the folders in best_students/ are used)
    <root>/tokenizer/STUDENT
Missing parts are skipped (e.g. a variant with only a teacher).

Run from the project folder (needs jsonDataset.py and modelSetups.py next to it), e.g.

  python evaluate_variants.py \\
      --variant DA_1layer=D:/models/run1  --variant DA_half=D:/models/run2  --variant noDA_half=D:/models/run3 \\
      --train-jsonl .../jigsaw_data_train.jsonl --test-jsonl .../jigsaw_data_test.jsonl \\
      --out results_variants --compare DA_half noDA_half

  --reuse      skip inference for models whose predictions are already saved in --out
  --plot-only  no inference at all: rebuild the table/plots from the saved predictions (works on a machine without models)

Metrics per model (all on the official test set; thresholds are tuned on the 10% validation split, same seed as the pipeline):
mean ROC-AUC (the competition metric), mean PR-AUC, macro F1 at 0.5, macro F1 with val-tuned thresholds, per-label AUC/F1.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SIZE_ORDER = {"S": 0, "M": 1, "L": 2}
LEADERBOARD_AUC = 0.98856


# ----------------------------------------------------------------------------- metrics
def best_threshold(p, y):
    from sklearn.metrics import precision_recall_curve
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def compute_metrics(Pv, Yv, Pt, Yt):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    out, auc_t, auc_v, pr_t, f05, ftun = {}, [], [], [], [], []
    for j, name in enumerate(LABELS):
        thr = best_threshold(Pv[:, j], Yv[:, j])
        a_t = roc_auc_score(Yt[:, j], Pt[:, j]) if 0 < Yt[:, j].sum() < len(Yt) else np.nan   # nan if a label has no positives (tiny test files)
        a_v = roc_auc_score(Yv[:, j], Pv[:, j]) if 0 < Yv[:, j].sum() < len(Yv) else np.nan
        p_t = average_precision_score(Yt[:, j], Pt[:, j])
        f_05 = f1_score(Yt[:, j], Pt[:, j] >= 0.5, zero_division=0)
        f_tn = f1_score(Yt[:, j], Pt[:, j] >= thr, zero_division=0)
        out.update({f"auc_{name}": a_t, f"f1_{name}": f_05, f"f1tuned_{name}": f_tn, f"thr_{name}": thr})
        auc_t.append(a_t); auc_v.append(a_v); pr_t.append(p_t); f05.append(f_05); ftun.append(f_tn)
    out.update(mean_auc_test=float(np.nanmean(auc_t)), mean_auc_val=float(np.nanmean(auc_v)), mean_pr_test=float(np.nanmean(pr_t)),
               macro_f1_05=float(np.mean(f05)), macro_f1_tuned=float(np.mean(ftun)))
    return out


# ----------------------------------------------------------------------------- inference
def predict(model, tok, texts, device, batch_size, max_length):
    import torch
    order = np.argsort([len(t) for t in texts])               # length-sorted batches: little padding
    probs = np.zeros((len(texts), 6), dtype=np.float32)
    model.eval().to(device)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            probs[idx] = torch.sigmoid(model(**enc).logits).float().cpu().numpy()
    model.cpu()
    return probs


def load_data(args, preds_dir):
    import torch
    from torch.utils.data import random_split
    import jsonDataset
    train_ds = jsonDataset.jigsawDataset(Path(args.train_jsonl))
    n_train = int(0.9 * len(train_ds))
    _, val_subset = random_split(train_ds, [n_train, len(train_ds) - n_train], generator=torch.Generator().manual_seed(args.seed))
    val_texts = [train_ds[i][0] for i in val_subset.indices]
    Y_val = np.array([train_ds.labels[i] for i in val_subset.indices])
    test_ds = jsonDataset.jigsawDataset(Path(args.test_jsonl))
    test_texts = [test_ds[i][0] for i in range(len(test_ds))]
    Y_test = np.array(test_ds.labels)
    np.save(preds_dir / "labels_val.npy", Y_val)
    np.save(preds_dir / "labels_test.npy", Y_test)
    print(f"validation split: {len(val_texts)} samples, test set: {len(test_texts)} samples", flush=True)
    return val_texts, test_texts


def run_predict(args, preds_dir):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import modelSetups

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_texts = test_texts = None

    def have(vdir, name):
        return args.reuse and (vdir / f"{name}_val.npy").exists() and (vdir / f"{name}_test.npy").exists()

    def do(vdir, name, model, tok, bs, ml):
        nonlocal val_texts, test_texts
        if val_texts is None:
            val_texts, test_texts = load_data(args, preds_dir)
        print(f"  predicting {name} ...", flush=True)
        np.save(vdir / f"{name}_val.npy", predict(model, tok, val_texts, device, bs, ml))
        np.save(vdir / f"{name}_test.npy", predict(model, tok, test_texts, device, bs, ml))

    for spec in args.variant:
        vname, root = spec.split("=", 1)
        root = Path(root)
        vdir = preds_dir / vname
        vdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== variant {vname}: {root}", flush=True)

        # teacher
        tdir = root / "teacher_TL"
        if (tdir / "MODEL").exists() and not have(vdir, "teacher"):
            tok_dir = tdir / "TOKENIZER"
            tok = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else "answerdotai/ModernBERT-base")
            model = AutoModelForSequenceClassification.from_pretrained(tdir / "MODEL", num_labels=6, problem_type="multi_label_classification")
            json.dump({"params": sum(p.numel() for p in model.parameters())}, open(vdir / "teacher_meta.json", "w"))
            do(vdir, "teacher", model, tok, args.teacher_batch, 512)
            del model
        elif not (tdir / "MODEL").exists():
            print("  (no teacher found)")

        # students
        sdir = root / "best_students"
        stok_dir = root / "tokenizer" / "STUDENT"
        if not sdir.exists() or not stok_dir.exists():
            print("  (no students / student tokenizer found)")
            continue
        names_file = root / "student_names.json"
        names = json.load(open(names_file))["names"] if names_file.exists() else sorted(p.name for p in sdir.iterdir() if (p / "model.safetensors").exists())
        stok = AutoTokenizer.from_pretrained(stok_dir)
        ml = stok.model_max_length if stok.model_max_length < 10_000 else 256
        for name in names:
            weights = sdir / name / "model.safetensors"
            if not weights.exists():
                print(f"  {name}: no weights, skipped")
                continue
            if have(vdir, name):
                continue
            try:
                model = modelSetups.create_student_model(name, vocab_size=len(stok))
                model.load_state_dict(load_file(str(weights), device="cpu"))
            except Exception as e:                                       # architecture in modelSetups differs from the checkpoint
                print(f"  {name}: could not load ({str(e).splitlines()[0][:150]}), skipped")
                continue
            json.dump({"params": sum(p.numel() for p in model.parameters()), "size_mb": weights.stat().st_size / 2 ** 20},
                      open(vdir / f"{name}_meta.json", "w"))
            do(vdir, name, model, stok, args.student_batch, ml)
            del model


# ----------------------------------------------------------------------------- analysis + plots
def parse_name(name):
    if name == "teacher":
        return "Teacher", "", ""
    p = name.split("_")
    return p[0], "_".join(p[1:-1]), p[-1]


def build_table(preds_dir):
    import pandas as pd
    Yv, Yt = np.load(preds_dir / "labels_val.npy"), np.load(preds_dir / "labels_test.npy")
    rows = []
    for vdir in sorted(p for p in preds_dir.iterdir() if p.is_dir()):
        for ft in sorted(vdir.glob("*_test.npy")):
            name = ft.name[:-len("_test.npy")]
            fv = vdir / f"{name}_val.npy"
            if not fv.exists():
                continue
            m = compute_metrics(np.load(fv), Yv, np.load(ft), Yt)
            fam, shape, size = parse_name(name)
            meta = json.load(open(vdir / f"{name}_meta.json")) if (vdir / f"{name}_meta.json").exists() else {}
            rows.append(dict(variant=vdir.name, model=name, family=fam, shape=shape, size=size,
                             params=meta.get("params", np.nan), size_mb=meta.get("size_mb", np.nan), **m))
    return pd.DataFrame(rows)


SHAPE_COLOR = {"DeepNarrow": "#2a78d6", "ShortWide": "#eb6834", "Bottleneck": "#1baf7a"}   # categorical slots 1-3 (validated all-pairs)
FAMILY_STYLE = {"MLP": dict(marker="o", ls="-"), "Trans": dict(marker="s", ls="--")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _handles():
    import matplotlib.pyplot as plt
    hs = [plt.Line2D([], [], color=c, lw=2, label=s) for s, c in SHAPE_COLOR.items()]
    return hs + [plt.Line2D([], [], color=MUTED, lw=1.5, label=f, **st) for f, st in FAMILY_STYLE.items()]


def fig_by_variant(df, col, ylabel, title, path):
    import matplotlib.pyplot as plt
    variants = list(df["variant"].unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(4.6 * len(variants), 4.6), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], variants):
        _style(ax)
        d = df[(df["variant"] == v)].dropna(subset=[col])
        for (fam, shape), g in d[d["family"] != "Teacher"].groupby(["family", "shape"]):
            g = g.sort_values("params")
            ax.plot(g["params"], g[col], color=SHAPE_COLOR.get(shape, "#888"), lw=2, ms=7, mec="white", mew=1.5, **FAMILY_STYLE.get(fam, dict(marker="^", ls="-")))
        t = d[d["family"] == "Teacher"]
        if len(t):
            ax.axhline(float(t[col].iloc[0]), color=INK, lw=1.2, ls=":")
            ax.text(ax.get_xlim()[0], float(t[col].iloc[0]), " teacher", va="bottom", color=INK, fontsize=9)
        ax.set_xscale("log"); ax.set_xlabel("student parameters (log)", color=INK)
        ax.set_title(v, loc="left", color=INK, fontsize=11)
    axes[0][0].set_ylabel(ylabel, color=INK)
    axes[0][-1].legend(handles=_handles(), frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_compare(df, a, b, col, path):
    import matplotlib.pyplot as plt
    pa, pb = df[df["variant"] == a].set_index("model")[col], df[df["variant"] == b].set_index("model")[col]
    d = (pa - pb).dropna().sort_values()
    if d.empty:
        print(f"--compare: no models in common between {a} and {b}")
        return
    meta = df.drop_duplicates("model").set_index("model")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.6))
    _style(ax); ax.grid(False, axis="y")
    ax.axvline(0, color=MUTED, lw=1)
    ax.scatter(d.values, range(len(d)), s=55, color=[SHAPE_COLOR.get(meta.loc[m, "shape"], INK) for m in d.index], edgecolor="white", linewidth=1, zorder=3)
    ax.set_yticks(range(len(d)), d.index, fontsize=8)
    ax.set_xlabel(f"{col}:  {a}  minus  {b}", color=INK)
    ax.set_title(f"Effect of {a} vs {b} (mean {d.mean():+.4f}, {int((d > 0).sum())}/{len(d)} models better)", loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print(f"\n{a} minus {b} on {col}: mean {d.mean():+.4f}; {int((d > 0).sum())} of {len(d)} models better with {a}")


def run_analysis(args, preds_dir, out):
    import pandas as pd
    df = build_table(preds_dir)
    if df.empty:
        raise SystemExit("no predictions found")
    df.to_csv(out / "all_results.csv", index=False)
    fig_by_variant(df, "mean_auc_test", "mean ROC-AUC (test)", "Ranking quality vs. size", out / "auc_vs_params.png")
    fig_by_variant(df, "macro_f1_tuned", "macro F1 (val-tuned thresholds)", "F1 with tuned thresholds vs. size", out / "f1tuned_vs_params.png")
    fig_by_variant(df, "macro_f1_05", "macro F1 at threshold 0.5", "F1 at 0.5 vs. size", out / "f1_05_vs_params.png")
    if args.compare:
        fig_compare(df, args.compare[0], args.compare[1], "mean_auc_test", out / "compare_auc.png")
        fig_compare(df, args.compare[0], args.compare[1], "macro_f1_tuned", out / "compare_f1tuned.png")
    pd.set_option("display.width", 200)
    cols = ["variant", "model", "params", "mean_auc_val", "mean_auc_test", "mean_pr_test", "macro_f1_05", "macro_f1_tuned"]
    print(f"\n(leaderboard winner: mean AUC {LEADERBOARD_AUC})")
    print(df.sort_values(["variant", "mean_auc_test"], ascending=[True, False])[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nSaved table and figures to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", default=[], help="NAME=ROOT (repeatable)")
    ap.add_argument("--train-jsonl", default="/home/dl08e26/DATA/jigsaw_data_train.jsonl")
    ap.add_argument("--test-jsonl", default="/home/dl08e26/DATA/jigsaw_data_test.jsonl")
    ap.add_argument("--out", default="results_variants")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    ap.add_argument("--teacher-batch", type=int, default=32)
    ap.add_argument("--student-batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); preds_dir = out / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        if not args.variant:
            ap.error("give at least one --variant NAME=ROOT (or use --plot-only)")
        run_predict(args, preds_dir)
    run_analysis(args, preds_dir, out)


if __name__ == "__main__":
    main()
