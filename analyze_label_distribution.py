import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_splits

# Labels in the dataset
LABELS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data_analysis"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # take labels from training set
    with args.train.open(encoding="utf-8", newline="") as data_file:
        examples = [
            [int(row[label]) for label in LABELS]
            for row in csv.DictReader(data_file)
        ]

    # take 90% of dataset (the training part)
    train_indices, _ = data_splits.jigsaw_train_validation_indices(
        len(examples), args.seed
    )

    # compute label prevalence in the training subset
    train_size = len(train_indices)
    counts = [
        sum(examples[index][label_index] for index in train_indices)
        for label_index in range(len(LABELS))
    ]
    prevalence = [count / train_size * 100 for count in counts]

    # compute prevalence of unlabeled examples in the training subset
    unlabeled_count = sum(
        not any(examples[index]) for index in train_indices
    )
    unlabeled_prevalence = unlabeled_count / train_size * 100

    # make a table of label prevalence
    rows = [
        {
            "label": label,
            "example_count": counts[index],
            "training_examples": train_size,
            "prevalence_percent": prevalence[index],
        }
        for index, label in enumerate(LABELS)
    ]
    # add a row for unlabeled examples
    rows.append(
        {
            "label": "no_label",
            "example_count": unlabeled_count,
            "training_examples": train_size,
            "prevalence_percent": unlabeled_prevalence,
        }
    )

    # save the table to CSV
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "label_distribution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # plotting
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barv(
        range(len(order)),
        [rows[index]["prevalence_percent"] for index in order],
        color="#2a78d6",
    )
    axis.set_yticks(
        range(len(order)),
        [
            "No labels" if rows[index]["label"] == "no_label" else rows[index]["label"]
            for index in order
        ],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Examples (%)")
    axis.set_title("Label distribution", loc="left")
    axis.grid(axis="x", color="#e4e3df", linewidth=0.8)
    axis.set_axisbelow(True)

    max_prevalence = max(row["prevalence_percent"] for row in rows)
    axis.set_xlim(0, max_prevalence * 1.25)
    for position, index in enumerate(order):
        axis.text(
            rows[index]["prevalence_percent"] + max_prevalence * 0.02,
            position,
            f"{rows[index]['example_count']:,}",
            va="center",
        )

    figure.tight_layout()
    chart_path = args.out / "label_distribution.png"
    figure.savefig(chart_path, dpi=200)
    plt.close(figure)

    print(f"Training examples: {train_size:,}")
    print(f"{'Label':<18} {'Examples':>10} {'Prevalence':>12}")
    for row in rows:
        print(
            f"{'No labels' if row['label'] == 'no_label' else row['label']:<18} "
            f"{row['example_count']:>10,} "
            f"{row['prevalence_percent']:>11.2f}%"
        )
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved chart: {chart_path}")


if __name__ == "__main__":
    main()
