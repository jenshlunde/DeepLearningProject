import torch
from torch.utils.data import Dataset
import json
from transformers import DataCollatorForLanguageModeling


def collate_messages(batch):
    return batch


class MlmCollator:
    def __init__(self, tokenizer, max_length=512, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )

    def __call__(self, batch):
        chunks = []

        for messages in batch:
            tokenized = self.tokenizer(
                "\n".join(messages),
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=True,
                return_special_tokens_mask=True,
            )

            for chunk_index in range(len(tokenized["input_ids"])):
                chunks.append(
                    {
                        key: value[chunk_index]
                        for key, value in tokenized.items()
                        if key in {"input_ids", "attention_mask", "special_tokens_mask"}
                    }
                )

        return self.data_collator(chunks)

class SupervisedCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, labels = zip(*batch)
        tokenized = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return tokenized, torch.stack(labels)


class DualSupervisedCollator:
    def __init__(self, teacher_tokenizer, student_tokenizer, max_length=512):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, labels = zip(*batch)
        teacher_inputs = self.teacher_tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        student_inputs = self.student_tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
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
                record = json.loads(line.decode("utf-8"))                                   #Load binary data to json
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

