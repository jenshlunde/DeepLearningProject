import pandas as pd
import json
from pathlib import Path

# open all the files
data_path = Path(r"C:\Users\jensh\Desktop\Kaggle_disord_data_v3\jigsaw-toxic-comment-classification-challenge\versions\1")
train_df = pd.read_csv(data_path / "train.csv")
test_text = pd.read_csv(data_path / "test.csv")
test_labels = pd.read_csv(data_path / "test_labels.csv")
test_df = test_text.merge(test_labels, on="id")
test_df = test_df[test_df["toxic"] != -1]  # drop the unlabelded(?) data


    
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] #labels

def write_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "text": row["comment_text"],
                "labels": {col: int(row[col]) for col in label_cols},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

train_path = data_path / "jigsaw_data_train.jsonl"
test_path = data_path / "jigsaw_data_test.jsonl"
combined_path = data_path / "jigsaw_data_combined.jsonl"

write_jsonl(train_df, train_path)
write_jsonl(test_df, test_path)


def concatenate_jsonl(first_path, second_path, output_path):
    with open(output_path, "w", encoding="utf-8") as output_file:
        for input_path in (first_path, second_path):
            with open(input_path, encoding="utf-8") as input_file:
                for line in input_file:
                    output_file.write(line)


concatenate_jsonl(train_path, test_path, combined_path)