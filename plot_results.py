
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("C:/Users/jensh/Desktop/DL_RESULTS/noDA/OUTPUT/")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

KEYS = {
    "macro_f1": [r"macro.*f1"],
    "micro_f1": [r"micro.*f1"],
    "subset_acc": [r"subset.*acc", r"exact.*match"],
    "params": [r"total.*param", r"num.*param", r"param"],
    "size_mb": [r"checkpoint.*mb", r"checkpoint.*size", r"size.*mb"],
    "latency": [r"latency.*per_sample.*(?:wo|without|no).*(?:load|data)", r"latency.*(?:wo|without|no).*(?:load|data)", r"latency.*model", r"latency"],
    "memory": [r"peak.*cuda.*alloc", r"peak.*cuda", r"peak.*(?:rss|mem)"],
}
LABEL_PAT = {l: (r"(?<!severe_)toxic" if l == "toxic" else l) for l in LABELS}

SHAPE_COLOR = {"DeepNarrow": "#2a78d6", "ShortWide": "#eb6834", "Bottleneck": "#1baf7a"}
FAMILY_STYLE = {"MLP": dict(marker="o", ls="-"), "Trans": dict(marker="s", ls="--")}
SIZE_ORDER = {"S": 0, "M": 1, "L": 2}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def flatten(d, prefix=""):
    flat = {}
    if isinstance(d, dict):
        for k, v in d.items():
            flat.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        flat[prefix[:-1]] = float(d)
    return flat


def find(flat, patterns):
    for p in patterns:
        for k, v in flat.items():
            if re.search(p, k.lower()):
                return k, v
    return None, np.nan


def find_label_f1(flat, lab):
    p = LABEL_PAT[lab]
    return find(flat, [rf"{p}.*f1", rf"f1.*{p}"])


def parse_name(name):
    parts = name.split("_")
    return parts[0], "_".join(parts[1:-1]), parts[-1]


def load_row(json_path, name):
    with open(json_path, encoding="utf-8") as source:
        flat = flatten(json.load(source))
    row, used = {"model": name}, {}
    row["family"], row["shape"], row["size"] = parse_name(name)
    for m, pats in KEYS.items():
        used[m], row[m] = find(flat, pats)
    for lab in LABELS:
        used[f"f1_{lab}"], row[f"f1_{lab}"] = find_label_f1(flat, lab)
    return row, used, flat


def load_teacher_row(json_path):
    row, used, flat = load_row(json_path, "Teacher_x_L")
    row.update({"model": "Teacher", "family": "Teacher", "shape": "Teacher", "size": ""})
    return row, used, flat


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def legend_handles():
    hs = [plt.Line2D([], [], color=c, lw=2, label=s) for s, c in SHAPE_COLOR.items()]
    hs += [plt.Line2D([], [], color=MUTED, lw=1.5, label=f, **st) for f, st in FAMILY_STYLE.items()]
    return hs


def line_by_group(ax, df, x, y, logx=True):
    for (fam, shape), g in df.groupby(["family", "shape"]):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], color=SHAPE_COLOR.get(shape, "#888"), lw=2, ms=7, mec="white", mew=1.5,
                **FAMILY_STYLE.get(fam, dict(marker="^", ls="-")))
    if logx:
        ax.set_xscale("log")


def fig_f1_vs_params(df, teacher, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax)
    line_by_group(ax, df, "params", "macro_f1")
    if teacher is not None and not np.isnan(teacher.get("macro_f1", np.nan)):
        ax.axhline(teacher["macro_f1"], color=INK, lw=1.2, ls=":")
        ax.text(ax.get_xlim()[0], teacher["macro_f1"], "  teacher", va="bottom", color=INK, fontsize=9)
    ax.set_xlabel("Student parameters (log scale)")
    ax.set_ylabel("Macro F1 on Jigsaw test")
    ax.set_title("Accuracy vs. model size", loc="left", color=INK, fontsize=12)
    ax.legend(handles=legend_handles(), frameon=False, fontsize=9, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_heatmap(df, teacher, path):
    d = df.sort_values(["family", "shape", "size"], key=lambda s: s.map(SIZE_ORDER).fillna(s) if s.name == "size" else s)
    M = d[[f"f1_{l}" for l in LABELS]].to_numpy(dtype=float)
    teacher_values = np.array([[teacher[f"f1_{l}"] for l in LABELS]]) if teacher is not None else np.empty((0, len(LABELS)))
    M = np.vstack([M, teacher_values])
    labels = list(d["model"]) + (["Teacher"] if teacher is not None else [])
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(0.01, np.nanmax(M)), aspect="auto")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, color=MUTED, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.6 * np.nanmax(M) else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Per-label F1 (threshold 0.5)", loc="left", color=INK, fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_pareto(df, teacher, path, xcol, xlabel, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax)
    for _, r in df.iterrows():
        ax.scatter(r[xcol], r["macro_f1"], s=70, color=SHAPE_COLOR.get(r["shape"], "#888"),
                   edgecolor="white", linewidth=1.5, zorder=3,
                   marker=FAMILY_STYLE.get(r["family"], dict(marker="^"))["marker"])
        ax.annotate(r["size"], (r[xcol], r["macro_f1"]), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=MUTED)
    if teacher is not None and not np.isnan(teacher.get("macro_f1", np.nan)):
        ax.axhline(teacher["macro_f1"], color=INK, lw=1.2, ls=":")
        if not np.isnan(teacher.get(xcol, np.nan)):
            ax.scatter(teacher[xcol], teacher["macro_f1"], s=85, color=INK, marker="*", zorder=4,
                       label="Teacher")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro F1 on Jigsaw test")
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.legend(handles=[h for h in legend_handles()], frameon=False, fontsize=9, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_costs(df, teacher, path):
    panels = [("size_mb", "Checkpoint size"), ("latency", "Latency (as logged)"), ("memory", "Peak memory (as logged)")]
    panels = [(c, t) for c, t in panels if df[c].notna().any()]
    if not panels:
        return
    d = df.sort_values(["family", "shape", "size"], key=lambda s: s.map(SIZE_ORDER).fillna(s) if s.name == "size" else s)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 0.3 * len(d) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (c, t) in zip(axes, panels):
        style_axes(ax)
        ax.scatter(d[c], range(len(d)), s=45, color=[SHAPE_COLOR.get(s, "#888") for s in d["shape"]],
                   edgecolor="white", linewidth=1, zorder=3)  # dots, not bars: bar length is meaningless on a log axis
        if teacher is not None and not np.isnan(teacher.get(c, np.nan)):
            ax.scatter(teacher[c], len(d), s=85, color=INK, marker="*", zorder=4)
        ax.set_xscale("log")
        labels = list(d["model"]) + (["Teacher"] if teacher is not None else [])
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(t, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--list-keys", action="store_true")
    ap.add_argument("--teacher-json", default=None)
    a = ap.parse_args()
    
    root = Path(a.out)
    if a.teacher_json is None and (root / "teacher_TL/evaluation/evaluation_results.json").exists():
        a.teacher_json = str(root / "teacher_TL/evaluation/evaluation_results.json")
    res = root / "results"
    res.mkdir(exist_ok=True)

    rows, first_used = [], None
    for f in sorted((root / "best_students").glob("*/evaluation/evaluation_results.json")):
        name = f.parent.parent.name
        row, used, flat = load_row(f, name)
        if a.list_keys:
            print("\n".join(f"{k} = {v}" for k, v in flat.items()))
            return
        rows.append(row)
        first_used = first_used or used
    if not rows:
        raise SystemExit("no evaluation_results.json found under best_students/*/evaluation/")
    df = pd.DataFrame(rows)

    print("Key resolved for each metric (check these are the numbers you mean):")
    for m, k in first_used.items():
        print(f"  {m:18s} <- {k}")
    print(f"\n{len(df)} models. Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0].to_string() or '  none'}")

    teacher = None
    if a.teacher_json:
        teacher, t_used, _ = load_teacher_row(a.teacher_json)
        print("\nTeacher keys:", {k: v for k, v in t_used.items() if k in ("macro_f1", "params", "latency", "memory")})
        print(f"Teacher: macro F1 {teacher['macro_f1']:.3f}, latency {teacher['latency']:.1f}")

    summary = pd.concat([df, pd.DataFrame([teacher])], ignore_index=True) if teacher is not None else df
    summary.to_csv(res / "summary.csv", index=False)

    fig_f1_vs_params(df, teacher, res / "f1_vs_params.png")
    fig_heatmap(df, teacher, res / "per_label_f1_heatmap.png")
    if df["latency"].notna().any():
        fig_pareto(df, teacher, res / "f1_vs_latency.png", "latency", "Latency (as logged, log scale)",
                   "Accuracy vs. latency")
    fig_costs(df, teacher, res / "costs.png")

    cols = ["model", "macro_f1", "micro_f1", "params", "size_mb", "latency", "memory"]
    print("\n" + summary.sort_values("macro_f1", ascending=False)[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nSaved to {res}")


if __name__ == "__main__":
    main()
