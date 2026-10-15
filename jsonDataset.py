from pyparsing import Each
import torch
from torch.utils.data import Dataset
import json
from transformers import DataCollatorForLanguageModeling
import random
from typing import Any, Dict, Mapping


def collate_messages(batch):
    return batch

class DistillationCollator:
    def __init__(self, tokenizer, max_length = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        texts, teacher_logits, labels = zip(*batch)
        tokenized = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        tokenized["teacher_logits"] = torch.stack(teacher_logits)
        tokenized["labels"] = torch.stack(labels)
        return tokenized


class TextCollator:
    """Tokenize conversation messages for classifier teacher inference."""

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        texts = [
            "\n".join(item) if isinstance(item, (tuple, list)) else item
            for item in batch
        ]
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


class DualDistillationCollator:
    def __init__(self, tokenizer, max_length = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        texts = []
        teacher_logits = []
        labels = []
        labeled = []

        for source, item in batch:
            if isinstance(item, Mapping):
                text = item["text"]
                logits = item.get("teacher_logits", item.get("logits"))
                item_labels = item.get("labels")
            elif isinstance(item, (tuple, list)) and len(item) == 3:
                text, logits, item_labels = item
            else:
                raise ValueError(
                    "Dual distillation items must contain text, teacher logits, "
                    "and optional labels. Raw datasets cannot provide teacher logits."
                )

            if logits is None:
                raise ValueError(
                    f"The {source} item has no teacher logits. "
                    "Compute classifier logits for both datasets before distillation."
                )

            if isinstance(text, (tuple, list)):
                text = "\n".join(text)
            texts.append(text)
            teacher_logits.append(torch.as_tensor(logits, dtype=torch.float32))

            is_labeled = source == "labeled" and item_labels is not None
            labeled.append(is_labeled)
            labels.append(
                torch.as_tensor(item_labels, dtype=torch.float32)
                if is_labeled
                else torch.zeros_like(teacher_logits[-1])
            )

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized["teacher_logits"] = torch.stack(teacher_logits)
        tokenized["labels"] = torch.stack(labels)
        tokenized["labeled_mask"] = torch.tensor(labeled, dtype=torch.bool)
        return tokenized

class MlmCollator:
    def __init__(self, tokenizer, max_length = 512, mlm_probability = 0.15, max_chunks = 1, 
                 random_start = True, chars_per_token_cap = 6):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_chunks = max_chunks
        self.random_start = random_start
        self.char_cap = max_length * max_chunks * chars_per_token_cap
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer = tokenizer, mlm = True, mlm_probability = mlm_probability,
        )
        
    def __call__(self, batch):
        chunks = []

        for messages in batch:
            start = random.randrange(len(messages)) if self.random_start else 0     #to avoid seeing the same text every epoch
            text = "\n".join(messages[start:])[:self.char_cap]                      #limits the number of characters to be tokenized
                        
            tokenized = self.tokenizer(text, truncation = True, max_length = self.max_length, return_overflowing_tokens = self.max_chunks > 1, return_special_tokens_mask = True)

            if self.max_chunks > 1:
                n_chunks = min(len(tokenized["input_ids"]), self.max_chunks)
                for i in range (n_chunks):
                    chunks.append({k: tokenized[k][i] for k in ("input_ids", "attention_mask", "special_tokens_mask")})
            else:
                chunks.append({k: tokenized[k] for k in ("input_ids", "attention_mask", "special_tokens_mask")})

        return self.data_collator(chunks)

class SupervisedCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, labels = zip(*batch)
        tokenized = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        tokenized["labels"] = torch.stack(labels)
        return tokenized


class DualSupervisedCollator:
    def __init__(self, teacher_tokenizer, student_tokenizer):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer

    def __call__(self, batch):
        texts, labels = zip(*batch)
        teacher_inputs = self.teacher_tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        student_inputs = self.student_tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        return teacher_inputs, student_inputs, torch.stack(labels)


class jigsawDataset(Dataset):
    def __init__(self, text_path):
        self.text_path = text_path
        self.offsets = []
        self.labels = []
        self.class_names = ( "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",)

        # Index the file 
        with open(self.text_path, "rb") as file:
            while True:
                offset = file.tell()        #Where am i in the file
                line = file.readline()      #Get the current text
                if not line:                #If end of file, break
                    break
                record = json.loads(line.decode("utf-8"))                                   #Load binary data to json
                self.offsets.append(offset)                                                 #save the offset for this line
                self.labels.append([record["labels"][name] for name in self.class_names])   #save the labels for this line
        self._file = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if self._file is None:
            self._file = open(self.text_path, "rb")
        self._file.seek(self.offsets[idx])
        record = json.loads(self._file.readline().decode("utf-8"))
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return record["text"], labels


class jef1056dataset(Dataset):
    def __init__(self, text_path):
        self.text_path = text_path
        self.offsets = []

        # Index the file 
        with open(self.text_path, "rb") as file:
            while True:
                offset = file.tell()        #Where am i in the file
                line = file.readline()      #Get the current text
                if not line:                #If end of file, break
                    break
                #record = json.loads(line.decode("utf-8"))        #NOT NEEDED AS CURRENTLY SETUP AND SLOW  #Load binary data to json
                self.offsets.append(offset)                                                 #save the offset for this line
        self._file = None
    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if self._file is None:
            self._file = open(self.text_path, "rb")

        self._file.seek(self.offsets[idx])
        record = json.loads(self._file.readline().decode("utf-8"))
        return record["messages"]

class DualDataset(Dataset):    #Alternate samples from jef1056 and jigsaw dataset.
    def __init__(self, dataset1, dataset2):
        if not len(dataset1) or not len(dataset2):
            raise ValueError("Both datasets must contain at least one item.")
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return 2 * max(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, idx):
        source_index = idx // 2
        if idx % 2 == 0:
            return "unlabeled", self.dataset1[source_index % len(self.dataset1)]
        return "labeled", self.dataset2[source_index % len(self.dataset2)]


if __name__ == "__main__":

    batch_size = 4

    dataset_jigsaw = jigsawDataset("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_combined.jsonl")
#    dataset_jef1056 = jef1056dataset("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/discord-v3-antispam_shortRun.jsonl")

#    jef1056_loader = torch.utils.data.DataLoader(
#        dataset_jef1056,
#        batch_size=batch_size,
#        shuffle=True,
#        num_workers=2,
#        collate_fn=collate_messages,
#    )

    jigsaw_loader = torch.utils.data.DataLoader(
        dataset_jigsaw,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
   

#    jef1056_dataiter = iter(jef1056_loader)
#    for i in range(2):
#        batch = next(jef1056_dataiter)
#        for convo in batch:
#            print(f"############# JEF1056 [{i}] ############  {convo}")


    jigsaw_dataiter = iter(jigsaw_loader)
    for i in range(2):
        batch = next(jigsaw_dataiter)
        for labels, message in batch: ##doens't work, gets all messages then all labels, need to fix this??
            print(f"### JIGSAW [{i}] MESSAGE ### \n {message}")
            print(f"### JIGSAW [{i}] LABEL  ### \n {labels}")