from collections.abc import Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time
import warnings
from pathlib import Path
from torch.utils.data import Dataset
from transformers import Trainer
import itertools


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.jit\.script` is deprecated.*",
        category=FutureWarning,
    )
    from torcheval.metrics.functional import (
        binary_f1_score,
        binary_precision,
        binary_recall,
        multilabel_accuracy,
    )

def eval_sigmoid_05(logits):
    return torch.sigmoid(logits) > 0.5


def compute_multilabel_metrics(prediction):
    """Metrics callback for Hugging Face Trainer.evaluate()."""
    logits = prediction.predictions
    labels = np.asarray(prediction.label_ids).astype(bool)
    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_tensor = torch.from_numpy(
        1 / (1 + np.exp(-np.asarray(logits)))
    )
    labels_tensor = torch.from_numpy(labels)
    f1 = torch.stack([
        binary_f1_score(predicted_tensor[:, index], labels_tensor[:, index])
        for index in range(labels_tensor.shape[1])
    ])
    precision = torch.stack([
        binary_precision(predicted_tensor[:, index], labels_tensor[:, index])
        for index in range(labels_tensor.shape[1])
    ])
    recall = torch.stack([
        binary_recall(predicted_tensor[:, index], labels_tensor[:, index])
        for index in range(labels_tensor.shape[1])
    ])
    label_names = (
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate",
    )

    metrics = {
        "subset_accuracy": float(multilabel_accuracy(
            predicted_tensor, labels_tensor, criteria="exact_match"
        )),
        "hamming_accuracy": float(multilabel_accuracy(
            predicted_tensor, labels_tensor, criteria="hamming"
        )),
        "macro_f1": float(f1.mean()),
        "micro_precision": float(binary_precision(
            predicted_tensor.flatten(), labels_tensor.flatten()
        )),
        "micro_recall": float(binary_recall(
            predicted_tensor.flatten(), labels_tensor.flatten()
        )),
        "micro_f1": float(binary_f1_score(
            predicted_tensor.flatten(), labels_tensor.flatten()
        )),
    }
    metrics.update({
        f"{name}_precision": float(precision[index])
        for index, name in enumerate(label_names)
    })
    metrics.update({
        f"{name}_recall": float(recall[index])
        for index, name in enumerate(label_names)
    })
    metrics.update({
        f"{name}_f1": float(f1[index])
        for index, name in enumerate(label_names)
    })
    return metrics


def train_model_unsupervised(model, optimizer, dataloader, device, loss_fn=None, prints=0):
    model.to(device)
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch, batch_data in enumerate(dataloader):
        labels = None

        if isinstance(batch_data, Mapping):
            model_inputs = {key: value.to(device) for key, value in batch_data.items()}
            labels = model_inputs.get("labels")
        else:
            model_inputs, labels = batch_data
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
            labels = labels.to(device)

        output = model(**model_inputs)

        if loss_fn is None:
            if not hasattr(output, "loss") or output.loss is None:
                raise ValueError("loss_fn is required when the model does not return a loss.")
            loss = output.loss
        else:
            if labels is None:
                raise ValueError("The dataloader must provide labels when loss_fn is used.")
            logits = output.logits if hasattr(output, "logits") else output
            loss = loss_fn(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        if batch == 0 or batch % 100 == 0 and prints >= 2:
            loss, current = loss.item(), (batch + 1) * batch_size
            print(f"US Train loss: [{current:>5d}/{len(dataloader.dataset):>5d}]") #print update
            
    avg_loss = total_loss / num_samples

    if prints >= 5:
        print(f"train_model_unsupervised - loss: {avg_loss:>7f}")

    return avg_loss


def validate_model_unsupervised(model, dataloader, device, loss_fn=None, prints=0):
    model.to(device)
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            if isinstance(batch_data, Mapping):
                model_inputs = {key: value.to(device) for key, value in batch_data.items()}
                labels = model_inputs.get("labels")
            else:
                model_inputs, labels = batch_data
                model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
                labels = labels.to(device)
                        
            output = model(**model_inputs)
            if loss_fn is None:
                if not hasattr(output, "loss") or output.loss is None:
                    raise ValueError(
                        "loss_fn is required when the model does not return a loss."
                    )
                loss = output.loss
            else:
                if labels is None:
                    raise ValueError("The dataloader must provide labels when loss_fn is used.")
                logits = output.logits if hasattr(output, "logits") else output
                loss = loss_fn(logits, labels)

            batch_size = next(value for value in model_inputs.values() if value.ndim > 0).size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            if batch == 0 or batch % 100 == 0 and prints >= 2:
                loss, current = loss.item(), (batch + 1) * batch_size
                print(f"US Val loss: [{current:>5d}/{len(dataloader.dataset):>5d}]") #print update

    avg_loss = total_loss / num_samples

    if prints >= 5:
        print(f"validate_model_unsupervised - loss: {avg_loss:>7f}")

    return avg_loss


def train_model_supervised(model, optimizer, loss_fn, dataloader, device, eval_func = eval_sigmoid_05, prints=0):
    model.to(device)                                            #Move model to device
    model.train()                                               #Set model to training mode
    total_loss = 0.0                        
    correct_total = 0
    correct_individual = 0                             
    num_samples = 0                   
    num_labels = 0

    for batch, batch_data in enumerate(dataloader):
        labels = None

        if isinstance(batch_data, Mapping):
            model_inputs = {key: value.to(device) for key, value in batch_data.items()}
            labels = model_inputs.get("labels")
        else:
            model_inputs, labels = batch_data
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
            labels = labels.to(device)

        output = model(**model_inputs)                                                #forward pass
        logits = output.logits
        loss = loss_fn(logits, labels)                                          #calc loss
        pred = eval_func(logits)                                                #pred
        optimizer.zero_grad()                                                   #Avoid accum gradients
        loss.backward()                                                         #backprop
        optimizer.step()                                                        #update model

        correct_total += (pred == labels.bool()).all(dim=1).sum().item()        #accum total corrects
        correct_individual += (pred == labels.bool()).sum().item()              #accum individual corrects
        
        batch_size = labels.size(0)                                             #accum samples
        total_loss += loss.item() * batch_size                                  #accum loss
        num_samples += batch_size    
        num_labels += labels.numel()
        
        if batch == 0 or batch % 100 == 0 and prints >= 2:
            loss, current = loss.item(), (batch + 1) * batch_size
            print(f"S Train loss: [{current:>5d}/{len(dataloader.dataset):>5d}]") #print update

    avg_loss = total_loss / num_samples                                         #bookkeeping
    avg_accuracy_total = correct_total / num_samples
    avg_accuracy_individual = correct_individual / num_labels

    if prints >= 5:
      print(f"train_model_supervised - loss: {avg_loss:>7f}, Accuracy: {(100*avg_accuracy_total):>0.1f}% (Total), {(100*avg_accuracy_individual):>0.1f}% (Individual)")

    return avg_loss, avg_accuracy_total, avg_accuracy_individual


def validate_model_supervised(model, loss_fn, dataloader, device, eval_func = eval_sigmoid_05, prints=0):
    model.to(device)                            # Move model to device
    model.eval()                                # Set model to eval
    val_total_loss = 0
    correct_total = 0
    correct_individual = 0                             
    num_samples = 0
    num_labels = 0

    with torch.no_grad():                       # no calc gradients
        for batch, (inputs, labels) in enumerate(dataloader):                       # Iterate over batches of data
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)
            
            output = model(**inputs)                                                #forward pass
            logits = output.logits                                                   #get logits
            pred = eval_func(logits)                                                 #get predictions
            val_loss = loss_fn(logits, labels)                                      #calc loss

            correct_total += (pred == labels.bool()).all(dim=1).sum().item()        #accum total corrects
            correct_individual += (pred == labels.bool()).sum().item()              #accum individual corrects
        
            batch_size = labels.size(0)                                             #accum samples
            val_total_loss += val_loss.item() * batch_size                          #accum loss
            num_samples += batch_size    
            num_labels += labels.numel()
        
            if batch == 0 or batch % 100 == 0 and prints >= 2:
                val_loss, current = val_loss.item(), (batch + 1) * batch_size
                print(f"S Val loss: [{current:>5d}/{len(dataloader.dataset):>5d}]") #print update

    #bookkeeping
    val_avg_loss = val_total_loss / num_samples
    val_avg_accuracy_total = correct_total / num_samples
    val_avg_accuracy_individual = correct_individual / num_labels

    if prints >= 5:
        print(f"validate_model_supervised - loss: {val_avg_loss:>8f}, Accuracy: {(100*val_avg_accuracy_total):>0.1f}% (Total), {(100*val_avg_accuracy_individual):>0.1f}% (Individual)")

    return val_avg_loss, val_avg_accuracy_total, val_avg_accuracy_individual


    
def test_model_supervised(model, loss_fn, dataloader, device, eval_func = eval_sigmoid_05, prints=0):
    model.to(device)                            # Move model to device
    model.eval()                                # Set model to eval
    test_total_loss = 0
    correct_total = 0
    correct_individual = 0                             
    num_samples = 0
    num_labels = 0

    with torch.no_grad():                       # no calc gradients
        for batch, (inputs, labels) in enumerate(dataloader):                       # Iterate over batches of data
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)
            
            output = model(**inputs)                                                # forward pass
            logits = output.logits                                                  # get logits
            pred = eval_func(logits)                                                # get predictions
            test_loss = loss_fn(logits, labels)                                     # calc loss

            correct_total += (pred == labels.bool()).all(dim=1).sum().item()        # accum corrects
            correct_individual += (pred == labels.bool()).sum().item()              # accum individual corrects
        
            batch_size = labels.size(0)                                             #accum samples
            test_total_loss += test_loss.item() * batch_size                        #accum loss
            num_samples += batch_size    
            num_labels += labels.numel()

            if batch == 0 or batch % 100 == 0 and prints >= 2:
                loss, current = test_loss.item(), (batch + 1) * batch_size
                print(f"S Test loss:  [{current:>5d}/{len(dataloader.dataset):>5d}]") #print update
   
    #bookkeeping
    test_avg_loss = test_total_loss / num_samples
    test_avg_accuracy_total = correct_total / num_samples
    test_avg_accuracy_individual = correct_individual / num_labels

    if prints >= 5:
        print(f"test_model_supervised - loss: {test_avg_loss:>8f}, Accuracy: {(100*test_avg_accuracy_total):>0.1f}% (Total), {(100*test_avg_accuracy_individual):>0.1f}% (Individual)")

    return test_avg_loss, test_avg_accuracy_total, test_avg_accuracy_individual


def _process_rss_bytes():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        if hasattr(__import__("ctypes"), "windll"):
            import ctypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            success = ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            return counters.WorkingSetSize if success else None
        return None


def _directory_size_bytes(path):
    if path is None or not Path(path).exists():
        return None
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())

def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def evaluate_multilabel_model(model: torch.nn.Module, dataset: Dataset, collator, device: torch.device, 
                              batch_size: int = 8, num_workers: int = 0, threshold: float = 0.5, 
                              label_names = None, model_path = None, warmup_batches: int = 2, max_batches = None):
  
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                             num_workers=num_workers, collate_fn=collator)
    model.to(device)    
    model.eval()        #Freeze model parameters and set to eval mode

    if device.type == "cuda":       #Clear GPU memory cache and reset peak memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    #Bookkeeping
    logits_all = []
    labels_all = []
    total_loss = 0.0
    total_samples = 0
    timed_batches = 0
    timed_samples = 0
    timed_seconds_w_data = 0.0
    timed_seconds_wo_data = 0.0
    peak_rss = _process_rss_bytes()
    data_iterator = iter(dataloader)

    with torch.no_grad():
        batch_index = 0
        while max_batches is None or batch_index < max_batches: #While more batches

            start_t = time.perf_counter()                         #Timer here to also include data loading time

            try:
                batch = next(data_iterator)
            except StopIteration:
                break

            labels = batch.pop("labels").to(device).float()
            model_inputs = {key: value.to(device) for key, value in batch.items()}

            model_t = time.perf_counter() - start_t
            synchronize(device)
            logits = model(**model_inputs).logits
            synchronize(device)
            elapsed_t_w_data  = time.perf_counter() - start_t
            elapsed_t_wo_data = elapsed_t_w_data - model_t

            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())
            loss = nnF.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            if batch_index >= warmup_batches:
                timed_batches += 1
                timed_samples += labels.size(0)
                timed_seconds_w_data += elapsed_t_w_data
                timed_seconds_wo_data += elapsed_t_wo_data
            current_rss = _process_rss_bytes()
            if current_rss is not None:
                peak_rss = max(peak_rss or current_rss, current_rss)
            batch_index += 1

    if total_samples == 0:
        raise ValueError("Cannot evaluate a model on an empty dataset.")

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).bool()
    predictions = torch.sigmoid(logits)
    names = label_names or [f"label_{index}" for index in range(labels.size(1))]
    per_label = {}
    for index, name in enumerate(names):
        per_label[name] = {
            "precision": float(binary_precision(predictions[:, index], labels[:, index], threshold=threshold)),
            "recall": float(binary_recall(predictions[:, index], labels[:, index], threshold=threshold)),
            "f1": float(binary_f1_score(predictions[:, index], labels[:, index], threshold=threshold)),
        }

    return {
        "loss": total_loss / total_samples,
        "subset_accuracy": float(multilabel_accuracy(predictions, labels, threshold=threshold, criteria="exact_match")),
        "hamming_accuracy": float(multilabel_accuracy(predictions, labels, threshold=threshold, criteria="hamming")),
        "macro_f1": float(torch.tensor([item["f1"] for item in per_label.values()]).mean()),
        "micro_precision": float(binary_precision(predictions.flatten(), labels.flatten(), threshold=threshold)),
        "micro_recall": float(binary_recall(predictions.flatten(), labels.flatten(), threshold=threshold)),
        "micro_f1": float(binary_f1_score(predictions.flatten(), labels.flatten(), threshold=threshold)),
        "per_label": per_label,
        "benchmark": {
            "batch_size": batch_size,
            "warmup_batches": min(warmup_batches, batch_index),
            "timed_batches": timed_batches,
            "latency_ms_per_batch_w_dataload": (timed_seconds_w_data / timed_batches * 1000 if timed_batches else None),
            "latency_ms_per_sample_w_dataload": (timed_seconds_w_data / timed_samples * 1000 if timed_samples else None),
            "throughput_samples_per_second_w_dataload": (timed_samples / timed_seconds_w_data if timed_seconds_w_data else None),
            "latency_ms_per_batch_wo_dataload": (timed_seconds_wo_data / timed_batches * 1000 if timed_batches else None),
            "latency_ms_per_sample_wo_dataload": (timed_seconds_wo_data / timed_samples * 1000 if timed_samples else None),
            "throughput_samples_per_second_wo_dataload": (timed_samples / timed_seconds_wo_data if timed_seconds_wo_data else None),
            "peak_cpu_rss_mb": peak_rss / (1024 ** 2) if peak_rss is not None else None,
            "peak_cuda_allocated_mb": (torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None),
            "peak_cuda_reserved_mb": (torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else None),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "checkpoint_size_mb": (
                checkpoint_size / (1024 ** 2)
                if (checkpoint_size := _directory_size_bytes(model_path)) is not None
                else None
            ),
            "tokenization_included": True,
            "num_workers": num_workers,
        },
    }

def train_model_distill(student_model, teacher_model, optimizer, dataloader, device, alpha, temperature, eval_func=eval_sigmoid_05, prints=0):
    student_model.to(device)
    student_model.train()
    teacher_model.to(device)
    teacher_model.eval()
    
    dis_total_loss = 0
    correct_total = 0
    correct_individual = 0                             
    num_samples = 0
    num_labels = 0

    for batch, (teacher_inputs, student_inputs, labels) in enumerate(dataloader):
        teacher_inputs = {key: value.to(device) for key, value in teacher_inputs.items()}
        student_inputs = {key: value.to(device) for key, value in student_inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(**teacher_inputs).logits
        student_logits = student_model(**student_inputs).logits
        student_pred = eval_func(student_logits)

        soft_targets = torch.sigmoid(teacher_logits / temperature)
        soft_loss = nnF.binary_cross_entropy_with_logits(student_logits / temperature, soft_targets) * (temperature ** 2)
        hard_loss = nnF.binary_cross_entropy_with_logits(student_logits, labels)
        total_loss = (alpha * hard_loss) + ((1 - alpha) * soft_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        correct_total += (student_pred == labels.bool()).all(dim=1).sum().item()        #accum total corrects
        correct_individual += (student_pred == labels.bool()).sum().item()              #accum individual corrects
        
        batch_size = labels.size(0)                                             #accum samples
        dis_total_loss += total_loss.item() * batch_size                        #accum loss
        num_samples += batch_size    
        num_labels += labels.numel()
        
        if batch  == 0 or batch % 100 == 0 and prints >= 2:
            loss, current = total_loss.item(), (batch + 1) * batch_size
            print(f"S Dist loss: [{current:>5d}/{len(dataloader.dataset):>5d}]") #print update

    student_avg_loss = dis_total_loss / num_samples                                         #bookkeeping
    student_avg_accuracy_total = correct_total / num_samples
    student_avg_accuracy_individual = correct_individual / num_labels

    if prints >= 5:
      print(f"train_model_distill - loss: {student_avg_loss:>7f}, Accuracy: {(100*student_avg_accuracy_total):>0.1f}% (Total), {(100*student_avg_accuracy_individual):>0.1f}% (Individual)")

    return student_avg_loss, student_avg_accuracy_total, student_avg_accuracy_individual


def validate_model_distillation(student_model, teacher_model, dataloader, device, alpha, temperature, eval_func=eval_sigmoid_05, prints=0):
    student_model.to(device)
    student_model.eval()
    teacher_model.to(device)
    teacher_model.eval()

    total_loss = 0.0
    correct_total = 0
    correct_individual = 0
    num_samples = 0
    num_labels = 0

    with torch.no_grad():
        for teacher_inputs, student_inputs, labels in dataloader:
            teacher_inputs = {key: value.to(device) for key, value in teacher_inputs.items()}
            student_inputs = {key: value.to(device) for key, value in student_inputs.items()}
            labels = labels.to(device)
            
            teacher_logits = teacher_model(**teacher_inputs).logits
            student_logits = student_model(**student_inputs).logits

            soft_targets = torch.sigmoid(teacher_logits / temperature)
            soft_loss = nnF.binary_cross_entropy_with_logits(student_logits / temperature, soft_targets) * (temperature ** 2)
            hard_loss = nnF.binary_cross_entropy_with_logits(student_logits, labels)
            loss = (alpha * hard_loss) + ((1 - alpha) * soft_loss)
            predictions = eval_func(student_logits)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_total += (predictions == labels.bool()).all(dim=1).sum().item()
            correct_individual += (predictions == labels.bool()).sum().item()
            num_samples += batch_size
            num_labels += labels.numel()


    avg_loss = total_loss / num_samples
    avg_accuracy_total = correct_total / num_samples
    avg_accuracy_individual = correct_individual / num_labels
    if prints >= 1:
        print(
            f"validate_model_distillation - loss: {avg_loss:>7f}, "
            f"Accuracy: {(100 * avg_accuracy_total):>0.1f}% (Total), "
            f"{(100 * avg_accuracy_individual):>0.1f}% (Individual)"
        )
    return avg_loss, avg_accuracy_total, avg_accuracy_individual

def freeze_layers(model: torch.nn.Module, freeze_embeddings: bool = True, keep_last: int = 0, verbose: bool = False, return_trainable: bool = False):
    
    backbone = getattr(model, "base_model", None)           # Hugging Face models use ``base_model``
    if backbone is None:
        backbone = getattr(model, "model", model)           # some custom models use ``model``

    embeddings = getattr(backbone, "embeddings", None)

    if freeze_embeddings:
        if embeddings is not None:
            for parameter in embeddings.parameters():
                parameter.requires_grad = False

    if keep_last is not None:
        if keep_last < 0:
            raise ValueError("keep_last must be non-negative.")
        layers = getattr(backbone, "layers", None)
        if layers is None:
            encoder = getattr(backbone, "encoder", None)
            layers = getattr(encoder, "layer", None)

        if layers is None: #Safety check
            raise ValueError("The model does not expose transformer layers through 'layers' or 'encoder.layer'.")

        for layer in list(layers)[:-keep_last or None]:
            # Freeze all layers except the last ``keep_last`` layers.
            for parameter in layer.parameters():
                parameter.requires_grad = False


    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}, Frozen parameters: {frozen_params}, Trainable: {(trainable_params / total_params) * 100:.2f}%")

    if return_trainable:
        trainable_params_list = filter(lambda p: p.requires_grad, model.parameters())
        return trainable_params_list
    else:
        return None

class CachedLogitsDataset(Dataset):
    def __init__(self, seed, texts, labels, logits):
        self.seed = seed
        self.texts = texts
        self.labels = labels
        self.logits = logits

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": self.labels[idx] if self.labels is not None else None,
            "logits": self.logits[idx]
        }

def cache_logits(model: torch.nn.Module, dataset: Dataset, collator, cache_path: Path, batch_size: int, device: torch.device, seed: int = 42) -> CachedLogitsDataset:    
    if cache_path.exists():                                 #If the data already exists, just load and return it
        print(f"Loading cached logits from {cache_path}")
        dataset_logits = torch.load(cache_path)
        return CachedLogitsDataset(dataset_logits["seed"], dataset_logits["texts"], dataset_logits["labels"], dataset_logits["logits"])

    texts = [] 
    labels = []

    for index in range(len(dataset)):
        item = dataset[index]
        if isinstance(item, Mapping):                       #If the item is a mapping (like the jigsaw dataset?), extract the text and labels
            texts.append(item["text"])
            labels.append(item.get("labels"))
        elif isinstance(item, tuple) and len(item) == 2:    #If the item is a tuple (like the jigsaw dataset?), extract the text and labels
            texts.append(item[0])
            labels.append(item[1])
        else:                                               #Otherwise, there's no label, like the jef1056 dataset
            texts.append(item)
            labels.append(None)

    if all(item_label is not None for item_label in labels):
        labels = torch.stack(labels)
    elif any(item_label is not None for item_label in labels):
        raise ValueError("A dataset must provide labels for every item or for none of them.")
    else:
        labels = None

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval().to(device)
    
    logits = []
    with torch.no_grad():                                   #Compute logits without tracking gradients
        for batch in dataloader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            output = model(**inputs)
            logits.append(output.logits.cpu())

    logits_cat = torch.cat(logits)                                                           #Combine all logits into a single tensor
    dataset_logits = {"seed": seed, "texts": texts, "labels": labels, "logits": logits_cat}  #Dict with seed, texts, labels, and logits

    cache_path.parent.mkdir(parents=True, exist_ok=True)    #Create the parent directory if it doesn't exist
    torch.save(dataset_logits, cache_path)                  #Save the dataset_logits dictionary to the specified cache_path
    return CachedLogitsDataset(seed, texts, labels, logits_cat)


class DistillationTrainer(Trainer):
    def __init__(self, *args, alpha: float = .5, temperature: float = 2., **kwargs):
        super().__init__(*args, **kwargs)                                               #Initialize the base Trainer class with args and kwargs
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model: torch.nn.Module, inputs: dict, return_outputs: bool = False, num_items_in_batch: int = None):
        teacher_logits = inputs.pop("teacher_logits")                                  #Extract teacher logits from inputs
        labels = inputs["labels"].float()                                              #Get labels and convert to float
        student_output = model(**inputs)                                                      #Forward pass through the student model
        student_logits = student_output.logits                                                #Get student logits

        t = self.temperature
        soft_targets = torch.sigmoid(teacher_logits / t)                                                #teacher targets
        soft_loss = nnF.binary_cross_entropy_with_logits(student_logits / t, soft_targets) * (t ** 2)   #Soft loss
        hard_loss = nnF.binary_cross_entropy_with_logits(student_logits, labels)                        #Hard loss
        total_loss = (self.alpha * hard_loss) + ((1 - self.alpha) * soft_loss)                          #Total loss

        if return_outputs:
            return (total_loss, student_output)  
        else:
            return total_loss


class DualDistillationTrainer(Trainer):
    def __init__(self, *args, alpha: float = .5, temperature: float = 2., **kwargs):
        super().__init__(*args, **kwargs)                                               #Initialize the base Trainer class with args and kwargs
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model: torch.nn.Module, inputs: dict, return_outputs: bool = False, num_items_in_batch: int = None):

        # Extract teacher logits, labels, and labeled mask from inputs
        teacher_logits = inputs.pop("teacher_logits")
        labels = inputs.pop("labels").float()
        labeled_mask = inputs.pop("labeled_mask").bool()

        # Forward pass through the student model
        student_output = model(**inputs)
        student_logits = student_output.logits

        # Compute the soft targets and per-sample soft loss
        t = self.temperature
        soft_targets = torch.sigmoid(teacher_logits / t)
        per_sample_soft_loss = nnF.binary_cross_entropy_with_logits(student_logits / t, soft_targets, reduction="none").mean(dim=1) * (t ** 2)

        # For labeled samples, add the hard loss according to alpha. 
        # Averaging these per-sample losses handles unlabeled-only, labeled-only, and mixed batches uniformly.
        per_sample_loss = per_sample_soft_loss.clone()
        if labeled_mask.any():                                                  #if any labeled data
            per_sample_hard_loss = nnF.binary_cross_entropy_with_logits(student_logits[labeled_mask],labels[labeled_mask],reduction="none",).mean(dim=1)
            per_sample_loss[labeled_mask] = ((1 - self.alpha) * per_sample_soft_loss[labeled_mask] + self.alpha * per_sample_hard_loss)

        total_loss = per_sample_loss.mean()

        return (total_loss, student_output) if return_outputs else total_loss



N_DISCORD_MESSAGES = 1_000_000          # cap so Discord doesn't drown out Jigsaw

def student_text_iterator(unlabeled_set, labeled_set):
    # Training splits only (no val leakage). Strip "Username: " and split bundled sends.
    discord = (part
               for convo in unlabeled_set
               for m in convo
               for part in m.split(": ", 1)[-1].split("\\n")   # literal backslash-n in your files
               if part.strip())
    yield from itertools.islice(discord, N_DISCORD_MESSAGES)
    for text, _ in labeled_set:
        yield text