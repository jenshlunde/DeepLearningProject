###########################
#####     IMPORTS     #####
###########################

import json                                                         #To read data saved as json files
import copy
import torch
import glob
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification
from torch.utils.data import random_split

import csv                                         

import jsonDataset                                                  #Homegrown dataset class for reading json files
import modelFunctions                                               #Homegrown functions for models: train, val, test, ...
import time                                                         #Timing functions to see how long it takes to train

def main():

    print("\n\n")
    print("#######################")
    print("### PHASE 0 : SETUP ###")
    print("#######################")
    print("\n\n")

    print("Imports done")

    ##########################
    #####     CHECKS     #####
    ##########################

    print("Starting Checks...")

    print("PyTorch version:", torch.__version__)

    print("Checking for CUDA availability")

    if torch.cuda.is_available(): 
        print("GPU available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name(0))
        device = torch.device('cuda:0')
        cuda = torch.device('cuda:0')
        print(torch.version.cuda)
        print(torch.cuda.current_device())
        print(torch.cuda.device(0))
        print(torch.cuda.device_count())
        t = torch.cuda.get_device_properties(0).total_memory
        print(f"Total memory: {t/1e9} GB")
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print(f"Reserved memory: {r/1e9} GB")
    else:
        device = torch.device('cpu')
        print("no cuda available")
        

    print("Checks done")

    #########################
    #####     SETUP     #####
    #########################

    print("Starting Setup...")

    benchmarking_only = False
#    HF_HUB_DISABLE_SYMLINKS_WARNING=1       #Hides warning about model using symlink but not supported on Windows


    #####     MDOELS & TOKENIZERS     #####

    ## Teacher model(s)
    TEACHER_BASE = "answerdotai/ModernBERT-base"
    MODEL_PATH = "C:/Users/jensh/Desktop/Kaggle_disord_data_v3/Models/teacher_domain_adapted"
    BEST_TEACHER_PATH = "C:/Users/jensh/Desktop/Kaggle_disord_data_v3/Models"
    BEST_TEACHER_DA_NAME = "best_teacher_da_model_state_dict.pth"
    BEST_TEACHER_TL_NAME = "best_teacher_tl_model_state_dict.pth"
    BEST_STUDENT_PATH = "C:/Users/jensh/Desktop/Kaggle_disord_data_v3/Models"



    teacher_mlm = AutoModelForMaskedLM.from_pretrained(TEACHER_BASE)
    print(f"Teacher model initialized: {TEACHER_BASE}")

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE)
    print(f"Teacher tokenizer initialized: {TEACHER_BASE}")


    ## Student model(s)
    STUDENT_BASE = "bert-base-uncased"
    student_model_1 = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_BASE,
        num_labels=6,
        problem_type="multi_label_classification",

    )
    print(f"Student model initialized: {STUDENT_BASE}")

    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE)                            ##?? should look more into tokenizers
    print(f"Student tokenizer initialized: {STUDENT_BASE}")


    #####     DATALOADERS     #####

    ## unlabeled dataloader
    jef1056_batch_size = 16
    jef1056_workers = 4

    jef1056_dataset = jsonDataset.jef1056dataset("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/discord-v3-antispam_short_1000.jsonl")     ##SHORT VERSION??!!??#

    jef_train_size = int(0.9 * len(jef1056_dataset))
    jef_val_size = len(jef1056_dataset) - jef_train_size

    jef_train, jef_val = random_split(
        jef1056_dataset,
        [jef_train_size, jef_val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print("jef train data set initialized, length:", len(jef_train))
    print("jef validation data set initialized, length:", len(jef_val))

    COLLATOR_LENGTH = 512
    mlm_collator_train = jsonDataset.MlmCollator(teacher_tokenizer, max_length=COLLATOR_LENGTH, max_chunks=1, random_start=True)
    mlm_collator_val   = jsonDataset.MlmCollator(teacher_tokenizer, max_length=COLLATOR_LENGTH, max_chunks=1, random_start=False)

    jef1056_train_loader = torch.utils.data.DataLoader(
        jef_train,
        batch_size=jef1056_batch_size,
        shuffle=True,
        num_workers=jef1056_workers, 
        persistent_workers=True,
        collate_fn=mlm_collator_train,
    )
    jef1056_val_loader = torch.utils.data.DataLoader(
        jef_val,
        batch_size=jef1056_batch_size,
        shuffle=False,
        num_workers=jef1056_workers, 
        persistent_workers=True,
        collate_fn=mlm_collator_val,
    )
    print("jef10561 data loaders initialized.")

    ## labeled dataloader
    jigsaw_batch_size = 16
    jigsaw_workers = 4

    jigsaw_train_val = jsonDataset.jigsawDataset("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_train_1000.jsonl")
    jigsaw_test = jsonDataset.jigsawDataset("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_test_1000.jsonl")

    jigsaw_train_size = int(0.9 * len(jigsaw_train_val))
    jigsaw_val_size = len(jigsaw_train_val) - jigsaw_train_size

    jigsaw_train, jigsaw_val = random_split(
        jigsaw_train_val,
        [jigsaw_train_size, jigsaw_val_size],
        generator=torch.Generator().manual_seed(42),
    )
        
    print("jigsaw_train data set initialized, length:", len(jigsaw_train))
    print("jigsaw_val data set initialized, length:", len(jigsaw_val))
    print("jigsaw_test data set initialized, length:", len(jigsaw_test))

    jigsaw_collator = jsonDataset.SupervisedCollator(teacher_tokenizer,max_length=COLLATOR_LENGTH)
    jigsaw_distill_collator = jsonDataset.DualSupervisedCollator(
        teacher_tokenizer, student_tokenizer, max_length=COLLATOR_LENGTH
    )
    jigsaw_train_loader = torch.utils.data.DataLoader(
        jigsaw_train,
        batch_size=jigsaw_batch_size,
        shuffle=True,
        num_workers=jigsaw_workers,
        persistent_workers=True,    
        collate_fn=jigsaw_collator,
    )
    jigsaw_val_loader = torch.utils.data.DataLoader(
        jigsaw_val,
        batch_size=jigsaw_batch_size,
        shuffle=False,
        num_workers=jigsaw_workers, 
        persistent_workers=True,
        collate_fn=jigsaw_collator,
    ) 
    jigsaw_test_loader = torch.utils.data.DataLoader(
        jigsaw_test,
        batch_size=jigsaw_batch_size,
        shuffle=False,
        num_workers=jigsaw_workers, 
        persistent_workers=True,
        collate_fn=jigsaw_collator,
    )
    jigsaw_distill_train_loader = torch.utils.data.DataLoader(
        jigsaw_train, 
        batch_size=jigsaw_batch_size, 
        shuffle=True,
        num_workers=jigsaw_workers, 
        persistent_workers=True,
        collate_fn=jigsaw_distill_collator,
    )
    jigsaw_distill_val_loader = torch.utils.data.DataLoader(
        jigsaw_val, 
        batch_size=jigsaw_batch_size, 
        shuffle=False,
        num_workers=jigsaw_workers, 
        persistent_workers=True,
        collate_fn=jigsaw_distill_collator,
    )
    print("jigsaw data loaders initialized.")


    ## loss-functions,  optimizers and hyperparameters
    loss_fn_teacher_mlm = None                          # model comes with its own loss function (?)
    loss_fn_teacher = torch.nn.BCEWithLogitsLoss()      #?
    loss_fn_student = torch.nn.BCEWithLogitsLoss()      #?

    #optimizer_teacher_mlm = torch.optim.AdamW(teacher_mlm.parameters(),lr=5e-5,)           #is further down so the frozen weights are not included in the optimizer
    optimizer_student = torch.optim.AdamW(student_model_1.parameters(),lr=5e-5,) 
    
    student_alpha = 0.5         #??
    student_temperature = 2.0   #??


    print("Setup done")



    ############################
    #####     TRAINING     #####
    ############################


    epochs_da = 20
    update_pr_da = 1
    prints_da = 4
    patience_da = 5
    best_loss_da = float('inf')
    best_epoch_da = 0
    best_teacher_da_model_state_dict = copy.deepcopy(teacher_mlm.state_dict())

    teacher_domain_train_losses = []
    teacher_domain_val_losses = []
    patience_counter_da = 0

    # Freeze some of the model to lighten work - embeddings and half of the encoder layers
    for param in teacher_mlm.model.embeddings.parameters():
        param.requires_grad = False

    num_layers = len(teacher_mlm.model.layers)
    num_to_freeze = num_layers // 2

    for layer in teacher_mlm.model.layers[:num_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    # Sanity check
    train_params = sum(p.numel() for p in teacher_mlm.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in teacher_mlm.parameters())
    print(f"Trainable: {train_params:,} / {total_params:,} ({100*train_params/total_params:.1f}%)")

    teacher_params_train = filter(lambda p: p.requires_grad, teacher_mlm.parameters())
    optimizer_teacher_mlm = torch.optim.AdamW(teacher_params_train, lr=5e-5,)

    if benchmarking_only:
        print("Benchmarking only, skipping training and testing.")
            
        print("\n\n")
        print("######################################################")
        print("### PHASE 1A : BENCHMARK THE TIME NEEDED FOR THINGS ###")
        print("######################################################")
        print("\n\n")

        modelFunctions.benchmark_phase(teacher_mlm, jef1056_train_loader, device, use_amp=False)
        modelFunctions.benchmark_phase(teacher_mlm, jef1056_train_loader, device, use_amp=True)   # compare fp16

        print("Phase 0 complete - benchmarking done. Stopping.")
        return


    print("\n\n")
    print("####################################################")
    print("### PHASE 1 : DOMAIN ADAPTATION OF TEACHER MODEL ###")
    print("####################################################")
    print("\n\n")

    time_stamps_DA_epochs = []
    time_stamps_DA_train = []
    time_stamps_DA_val = []

    print("Starting domain adaptation training for teacher model... \n")


    for epoch in range(epochs_da):
        
        if prints_da >= 1:
            if epoch == 0 or epoch % update_pr_da == 0:
                print(f"--- Domain Adaption - Epoch {epoch + 1}/{epochs_da} ---")
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()

        domain_adaption_results = modelFunctions.train_model_unsupervised(
            model = teacher_mlm,
            dataloader = jef1056_train_loader,
            optimizer = optimizer_teacher_mlm,
            device = device,
            prints = prints_da,
        )
        teacher_domain_train_loss = domain_adaption_results
        teacher_domain_train_losses.append(teacher_domain_train_loss)

        if device.type == "cuda":
            torch.cuda.synchronize() 
        time_stamps_DA_train.append(time.perf_counter() - train_start)
        val_start = time.perf_counter()
        
        # Validate the model for current epoch
        teacher_domain_val = modelFunctions.validate_model_unsupervised(
            model = teacher_mlm,
            dataloader = jef1056_val_loader,
            device = device,
            prints = prints_da,
        )
        teacher_domain_val_losses.append(teacher_domain_val)

        if device.type == "cuda":
            torch.cuda.synchronize()
        time_stamps_DA_val.append(time.perf_counter() - val_start)

        # Print results for current epoch if wanted
        if prints_da >= 3:
            print(f"Domain Adaption - Train loss: {teacher_domain_train_loss:.4f}")
            print(f"Domain Adaption - Val loss: {teacher_domain_val:.4f}")

        # Check for early stopping based on validation loss
        if teacher_domain_val < best_loss_da:
            best_loss_da = teacher_domain_val
            best_epoch_da = epoch
            best_teacher_da_model_state_dict = copy.deepcopy(teacher_mlm.state_dict())
            patience_counter_da = 0        
        else:
            patience_counter_da += 1
            if prints_da >= 2:
                print(f"Domain Adaption - No improvement in val for {patience_counter_da}/{patience_da} epochs.")

        if patience_counter_da >= patience_da:
            if prints_da >= 1:
                print(f"Domain Adaption - Early stopping at epoch {epoch}")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        time_stamps_DA_epochs.append(time.perf_counter() - epoch_start)

    torch.save(best_teacher_da_model_state_dict, f"{BEST_TEACHER_PATH}/{BEST_TEACHER_DA_NAME}")  # Save best model
    print(f"Best DA teacher model saved at epoch {best_epoch_da} with val loss: {best_loss_da:.4f}, saved at path: {BEST_TEACHER_PATH}/{BEST_TEACHER_DA_NAME}")


    print("Domain adaption done.")
    
    teacher_mlm.load_state_dict(best_teacher_da_model_state_dict)
    ##?? What is the difference between saving the model and saving the state_dict??? 
    teacher_mlm.save_pretrained(f"{MODEL_PATH}/teacher_domain_adapted")
    teacher_tokenizer.save_pretrained(f"{MODEL_PATH}/teacher_domain_adapted")
    print("Domain adapted Teacher model saved.")



    print("\n\n")
    print("####################################################")
    print("### PHASE 2 : TRANSFER LEARNING OF TEACHER MODEL ###")
    print("####################################################")
    print("\n\n")

    print("Re-initializing teacher model for transfer learning.")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        f"{MODEL_PATH}/teacher_domain_adapted",
        num_labels=6,
        problem_type="multi_label_classification",
    )

    epochs_tl = 20
    update_pr_tl = 1
    patience_tl = 5
    best_loss_tl = float('inf')
    best_epoch_tl = 0
    prints_tl = 4
    best_teacher_tl_model_state_dict = copy.deepcopy(teacher_model.state_dict())

    teacher_transfer_train_losses = []
    teacher_transfer_train_accuracies_total = []
    teacher_transfer_train_accuracies_individual = []
    teacher_transfer_val_losses = []
    teacher_transfer_val_accuracies_total = []
    teacher_transfer_val_accuracies_individual = []
    patience_counter_tl = 0

    # Freeze some of the model to lighten work - embeddings and all layers except the last three (?)
    for param in teacher_model.model.embeddings.parameters():
        param.requires_grad = False

    num_to_freeze = len(teacher_model.model.layers) - 3         # Freeze all but the last three layers
    for layer in teacher_model.model.layers[:num_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    # Sanity check
    train_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"Trainable: {train_params:,} / {total_params:,} ({100*train_params/total_params:.1f}%)")

    teacher_params_train = filter(lambda p: p.requires_grad, teacher_model.parameters())
    optimizer_teacher = torch.optim.AdamW(teacher_params_train, lr=5e-5,)

    print("Teacher model re-initialized for transfer learning.")

    print("Starting transfer learning for teacher model... \n")

    time_stamps_TL_epochs = []
    time_stamps_TL_train = []
    time_stamps_TL_val = []

    for epoch in range(epochs_tl):
        
        if prints_tl >= 1:
            if epoch % update_pr_tl == 0:
                print(f"--- Transfer Learning - Epoch {epoch + 1}/{epochs_tl} ---")

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        
        # Train the model for current epoch
        teacher_transfer_train = modelFunctions.train_model_supervised(
            model = teacher_model,
            optimizer = optimizer_teacher,
            dataloader = jigsaw_train_loader,
            device = device,
            loss_fn = loss_fn_teacher,
            eval_func = modelFunctions.eval_sigmoid_05,
            prints = prints_tl,
        )
        #Save results from training
        train_loss, train_accuracy_total, train_accuracy_individual = teacher_transfer_train
        teacher_transfer_train_losses.append(train_loss)
        teacher_transfer_train_accuracies_total.append(train_accuracy_total)
        teacher_transfer_train_accuracies_individual.append(train_accuracy_individual)

        if device.type == "cuda":
            torch.cuda.synchronize()
        time_stamps_TL_train.append(time.perf_counter() - train_start)
        val_start = time.perf_counter()

        # Validate the model for current epoch
        teacher_transfer_val = modelFunctions.validate_model_supervised(
            model = teacher_model,
            dataloader = jigsaw_val_loader,
            device = device,
            loss_fn = loss_fn_teacher,
            eval_func = modelFunctions.eval_sigmoid_05,
            prints = prints_tl,
        )
        #Save results from validation
        val_loss, val_accuracy_total, val_accuracy_individual = teacher_transfer_val
        teacher_transfer_val_losses.append(val_loss)
        teacher_transfer_val_accuracies_total.append(val_accuracy_total)
        teacher_transfer_val_accuracies_individual.append(val_accuracy_individual)

        if device.type == "cuda":
            torch.cuda.synchronize()
        time_stamps_TL_val.append(time.perf_counter() - val_start)

        # Print results for current epoch if wanted
        if prints_tl >= 3:
            print(f"Transfer Learning - Train loss: {train_loss:.4f}, Train accuracy (total): {train_accuracy_total:.4f}, Train accuracy (individual): {train_accuracy_individual:.4f}")
            print(f"Transfer Learning - Val loss: {val_loss:.4f}, Val accuracy (total): {val_accuracy_total:.4f}, Val accuracy (individual): {val_accuracy_individual:.4f}")

        # Check for early stopping based on validation loss
        if val_loss < best_loss_tl:
            best_loss_tl = val_loss
            best_epoch_tl = epoch
            best_teacher_tl_model_state_dict = copy.deepcopy(teacher_model.state_dict())  # Save the best model's state_dict
            patience_counter_tl = 0
        else:
            patience_counter_tl += 1
            if prints_tl >= 2:
                print(f"Transfer Learning - No improvement in val for {patience_counter_tl}/{patience_tl} epochs.")

        if patience_counter_tl >= patience_tl:
            if prints_tl >= 1:
                print(f"Transfer Learning - Early stopping at epoch {epoch}")
            break

        if prints_tl >= 1 and epoch % update_pr_tl == 0:
            print(f"Best TL epoch: {best_epoch_tl}, validation loss: {best_loss_tl:.4f}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        time_stamps_TL_epochs.append(time.perf_counter() - epoch_start)

    torch.save(best_teacher_tl_model_state_dict, f"{BEST_TEACHER_PATH}/{BEST_TEACHER_TL_NAME}")  # Saving best model
    print(f"Best TL teacher model saved at epoch {best_epoch_tl} with val loss: {best_loss_tl:.4f}, saved at path: {BEST_TEACHER_PATH}/{BEST_TEACHER_TL_NAME}")



    print("\n\n")
    print("##########################################")
    print("### PHASE 3 : DISTILLING ONTO STUDENTS ###")
    print("##########################################")
    print("\n\n")

    #Load best teacher model
    teacher_model.load_state_dict(torch.load(f"{BEST_TEACHER_PATH}/{BEST_TEACHER_TL_NAME}", weights_only=True))  #only weights???

    print("Starting distilling of teacher model to student model...")
    epochs_dis = 20
    update_pr_dis = 1
    patience_dis = 5
    best_loss_dis = float('inf')
    best_epoch_dis = 0
    prints_dis = 4

    student_distil_losses = []
    student_distil_accuracies_total = []
    student_distil_accuracies_individual = []
    patience_counter_dis = 0
    best_student_model_1_state_dict = copy.deepcopy(student_model_1.state_dict())


    time_stamps_DIS_epochs = []
    time_stamps_DIS_train = []
    time_stamps_DIS_val  = []

    for epoch in range(epochs_dis):
        
        if prints_dis >= 1:
            if epoch % update_pr_dis == 0:
                print(f"--- Student Distilling - Epoch {epoch + 1}/{epochs_dis} ---")

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()

        # Distil the teacher onto the student for current epoch
        train_model_distill_results =  modelFunctions.train_model_distill(
            student_model = student_model_1, 
            teacher_model = teacher_model, 
            optimizer = optimizer_student, 
            dataloader = jigsaw_distill_train_loader, 
            device = device, 
            alpha = student_alpha, 
            temperature = student_temperature, 
            eval_func = modelFunctions.eval_sigmoid_05, 
            prints=prints_dis)
        student_train_loss, student_train_accuracy_total, student_train_accuracy_individual = train_model_distill_results

        student_distil_losses.append(student_train_loss)
        student_distil_accuracies_total.append(student_train_accuracy_total)
        student_distil_accuracies_individual.append(student_train_accuracy_individual)

        if device.type == "cuda":
            torch.cuda.synchronize()
        time_stamps_DIS_train.append(time.perf_counter() - train_start)
        val_start = time.perf_counter()

        val_model_distill_results = modelFunctions.validate_model_distillation(
            student_model=student_model_1,
            teacher_model=teacher_model,
            dataloader=jigsaw_distill_val_loader,
            device=device,
            alpha=student_alpha,
            temperature=student_temperature,
            eval_func=modelFunctions.eval_sigmoid_05,
            prints=prints_dis,
        )
        student_val_loss, student_val_accuracy_total, student_val_accuracy_individual = val_model_distill_results

        if device.type == "cuda":
            torch.cuda.synchronize()
        time_stamps_DIS_val.append(time.perf_counter() - val_start)

        if student_val_loss < best_loss_dis:
            best_loss_dis = student_val_loss
            best_epoch_dis = epoch
            best_student_model_1_state_dict = copy.deepcopy(student_model_1.state_dict())
            patience_counter_dis = 0
        else:
            patience_counter_dis += 1
            if prints_dis >= 2:
                print(f"Student Distilling - No improvement in val for {patience_counter_dis}/{patience_dis} epochs.")

        if patience_counter_dis >= patience_dis:
            if prints_dis >= 1:
                print(f"Student Distilling - Early stopping at epoch {epoch}")
            break

        # Print results for current epoch if wanted
        if prints_dis >= 3:
            print(f"Student Distilling - Train loss: {student_train_loss:.4f}, Train accuracy (total): {student_train_accuracy_total:.4f}, Train accuracy (individual): {student_train_accuracy_individual:.4f}")
            print(f"Student Distilling - Val loss: {student_val_loss:.4f}, Val accuracy (total): {student_val_accuracy_total:.4f}, Val accuracy (individual): {student_val_accuracy_individual:.4f}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        time_stamps_DIS_epochs.append(time.perf_counter() - epoch_start)


    torch.save(best_student_model_1_state_dict, f"{BEST_STUDENT_PATH}/best_student_model_1_state_dict.pth")
    print(f"Best student model saved at epoch {best_epoch_dis} with val loss: {best_loss_dis:.4f}")


    print("\n\n")
    print("################################")
    print("### PHASE 4 : TESTING MODELS ###")
    print("################################")
    print("\n\n")

    ## Test teacher model on labeled data
    ## Test student model(s) on labeled data


    timing_rows = []

    for phase, stages in {
        "DA": {
            "epochs": time_stamps_DA_epochs,
            "train": time_stamps_DA_train,
            "validation": time_stamps_DA_val,
        },
        "TL": {
            "epochs": time_stamps_TL_epochs,
            "train": time_stamps_TL_train,
            "validation": time_stamps_TL_val,
        },
        "DIS": {
            "epochs": time_stamps_DIS_epochs,
            "train": time_stamps_DIS_train,
            "validation": time_stamps_DIS_val,
        },
    }.items():
        for stage, durations in stages.items():
            for epoch, duration in enumerate(durations, start=1):
                timing_rows.append({
                    "phase": phase,
                    "epoch": epoch,
                    "stage": stage,
                    "duration_seconds": duration,
                })

    with open("time_stamps.csv", "w", newline="") as filehandle:
        writer = csv.DictWriter(
            filehandle,
            fieldnames=["phase", "epoch", "stage", "duration_seconds"],
        )
        writer.writeheader()
        writer.writerows(timing_rows)


#### TODO: 

# Save results from training, validation and testing for all models in a structured way (e.g., CSV, JSON, or a database) for later analysis and comparison.
# 
# DO MORE STUDENT MODELS, DIFFERENT ARCHITECTURES, HYPERPARAMETERS, ETC.
# DO PHASE 3.5: STUDENT MODEL FINE-TUNING ??
# DO PHASE 4: TESTING MODELS
# - TEST THE FINAL TEACHER ON JIGSAW TEST SET
# - TEST EVERY STUDENT ON THE JIGSAW TEST SET
# - RECORD METRICS FOR EACH STUDENT MODEL
# - - model name, parameter count, checkpoint size, test metrics, inference latency, peak memory usage, tokenizer, sequence length, thresholds
#
# Training controls 
# - learning rate scheduler
# - gradient clipping 
# - mixed precision training
# - checkpointing
# - reproducible random seed
#
# TUNE THRESHOLDS ON VALIDATION SET FOR EACH LABEL INSTEAD OF USING 0.5
# 
# BETTER METRICS THAN JUST ACCURACY:
# PR LABEL
# Precision
# Recall
# F1
# Macro F1
# Micro F1
# PR-AUC 
# - probably % correct labeled when NOT 0 (pr label and total)
#  
#Plots could include:
#Training and validation loss
#Training and validation F1
#Per-label F1
#Precision-recall curves
#Accuracy/F1 versus parameter count
#Accuracy/F1 versus latency
#Memory usage comparison
#Model size comparison

#Refactor for the sweep. Wrap the phases in functions taking a config dict and writing a results JSON per run, and add a global seed. That matters more than any single fix above once you have many students.

if __name__ == "__main__":
    main()

