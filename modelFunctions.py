from collections.abc import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time

def eval_sigmoid_05(logits):
    return torch.sigmoid(logits) > 0.5


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

def benchmark_phase(model, dataloader, device, warmup=5, measure=30, lr=5e-5, use_amp=False):
    """Time real training steps, then restore the model's weights so the benchmark leaves no trace."""
    model.to(device)
    model.train()
    backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    data_t, step_t, real_tok, padded_tok, seqs = 0.0, 0.0, 0, 0, 0
    if len(dataloader) == 0:
        raise ValueError("benchmark_phase requires a non-empty dataloader")
    it = iter(dataloader)

    for i in range(warmup + measure):
        t0 = time.perf_counter()
        try:
            batch = next(it)                               # includes tokenization + masking in the collator
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        if cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = model(**batch).loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if cuda: torch.cuda.synchronize()
        t2 = time.perf_counter()

        if i >= warmup:                                    # skip warmup: CUDA init, cudnn autotune, etc.
            data_t += t1 - t0
            step_t += t2 - t1
            real_tok += batch["attention_mask"].sum().item()
            padded_tok += batch["input_ids"].numel()
            seqs += batch["input_ids"].size(0)

    model.load_state_dict(backup)                          # undo the benchmark's weight updates

    per_batch = (data_t + step_t) / measure
    n_batches = len(dataloader)
    print(f"data loading : {1000 * data_t / measure:8.1f} ms/batch")
    print(f"compute      : {1000 * step_t / measure:8.1f} ms/batch")
    print(f"total        : {1000 * per_batch:8.1f} ms/batch  ({seqs / measure:.1f} seqs/batch)")
    print(f"throughput   : {padded_tok / (data_t + step_t):,.0f} padded tok/s "
          f"({real_tok / (data_t + step_t):,.0f} real tok/s, {100 * real_tok / padded_tok:.0f}% non-padding)")
    if cuda:
        print(f"peak VRAM    : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"estimated full epoch: {per_batch * n_batches / 3600:.2f} h  ({n_batches:,} batches)")


    