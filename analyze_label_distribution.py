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
    # set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data_analysis"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # open training dataset and take each label for each row and turn into integers
    with args.train.open(encoding="utf-8", newline="") as data_file:
        examples = []
        for row in csv.DictReader(data_file):
            label_values = [int(row[label]) for label in LABELS]
            examples.append(label_values)

    # take 90% of dataset (the training part)
    train_indices, _ = data_splits.jigsaw_train_validation_indices(
        len(examples), args.seed
    )

    # compute label prevalence in the training subset
    train_size = len(train_indices)
    counts = []
    for label_index in range(len(LABELS)):
        count = 0
        for index in train_indices:
            count += examples[index][label_index]
        counts.append(count)

    prevalence = []
    for count in counts:
        prevalence.append(count / train_size * 100)

    # compute prevalence of unlabeled examples in the training subset
    unlabeled_count = 0
    for index in train_indices:
        has_label = False
        for label_value in examples[index]:
            if label_value == 1:
                has_label = True
                break
        if not has_label:
            unlabeled_count += 1

    unlabeled_prevalence = unlabeled_count / train_size * 100

    # make a list of comments with labels and their count and prevalence
    rows = []
    for index, label in enumerate(LABELS):
        row = {
            "label": label,
            "example_count": counts[index],
            "prevalence_percent": prevalence[index],
        }
        rows.append(row)

    # make a list of comments without labels and their count and prevalence
    unlabeled_row = {
        "label": "no_label",
        "example_count": unlabeled_count,
        "prevalence_percent": unlabeled_prevalence,
    }
    rows.append(unlabeled_row)

    # write rows to csv file. Create new if it does not already exist
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "label_distribution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # plotting
    prevalence_values = []
    tick_labels = []
    for index in range(len(rows)):
        prevalence_values.append(rows[index]["prevalence_percent"])
        if rows[index]["label"] == "no_label":
            tick_labels.append("unlabeled comments")
        else:
            tick_labels.append(rows[index]["label"])

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        range(len(rows)),
        prevalence_values,
        color="lightblue",
    )
    axis.set_xticks(range(len(rows)), tick_labels, rotation=45)
    axis.set_ylabel("Percentage of labeled and unlabeled comments (%)")
    axis.set_title("Label distribution of comments")

    axis.set_ylim(0, 100) 
    for position, index in enumerate(range(len(rows))):
        axis.text(
            position,
            rows[index]["prevalence_percent"] + 1,
            f"{rows[index]['example_count']:,}",
            ha="center",
        )

    figure.tight_layout()

    # save plot in fodler
    chart_path = args.out / "label_distribution.png"
    figure.savefig(chart_path, dpi=200)
    plt.close(figure)

    print(f"Saved chart: {chart_path}")


if __name__ == "__main__":
    main()
