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

import jsonDataset                                                  #Homegrown dataset class for reading json files
import modelFunctions                                               #Homegrown functions for models: train, val, test, ...


#def main():

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

HF_HUB_DISABLE_SYMLINKS_WARNING=1       #Hides warning about model using symlink but not supported on Windows


#####     MDOELS & TOKENIZERS     #####

## Teacher model(s)
TEACHER_BASE = "answerdotai/ModernBERT-base"
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
jef1056_batch_size = 4
jef1056_workers = 0

jef1056_dataset = jsonDataset.jef1056dataset("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/discord-v3-antispam_short_100.jsonl")     ##SHORT VERSION??!!??#

jef_train_size = int(0.9 * len(jef1056_dataset))
jef_val_size = len(jef1056_dataset) - jef_train_size

jef_train, jef_val = random_split(
    jef1056_dataset,
    [jef_train_size, jef_val_size],
    generator=torch.Generator().manual_seed(42),
)
print("jef train data set initialized, length:", len(jef_train))
print("jef validation data set initialized, length:", len(jef_val))

mlm_collator = jsonDataset.MlmCollator(teacher_tokenizer, max_length=512)
jef1056_train_loader = torch.utils.data.DataLoader(
    jef_train,
    batch_size=jef1056_batch_size,
    shuffle=True,
    num_workers=jef1056_workers,
    collate_fn=mlm_collator,
)
jef1056_val_loader = torch.utils.data.DataLoader(
    jef_val,
    batch_size=jef1056_batch_size,
    shuffle=False,
    num_workers=jef1056_workers,
    collate_fn=mlm_collator,
)
print("jef10561 data loaders initialized.")

## labeled dataloader
jigsaw_batch_size = 4
jigsaw_workers = 0

jigsaw_train_val = jsonDataset.jigsawDataset("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_train_short.jsonl")
jigsaw_test = jsonDataset.jigsawDataset("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_test_short.jsonl")

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

jigsaw_collator = jsonDataset.SupervisedCollator(teacher_tokenizer,max_length=512)
jigsaw_distill_collator = jsonDataset.DualSupervisedCollator(
    teacher_tokenizer, student_tokenizer, max_length=512
)
jigsaw_train_loader = torch.utils.data.DataLoader(
    jigsaw_train,
    batch_size=jigsaw_batch_size,
    shuffle=True,
    num_workers=jigsaw_workers,
    collate_fn=jigsaw_collator,
)
jigsaw_val_loader = torch.utils.data.DataLoader(
    jigsaw_val,
    batch_size=jigsaw_batch_size,
    shuffle=False,
    num_workers=jigsaw_workers,
    collate_fn=jigsaw_collator,
) 
jigsaw_test_loader = torch.utils.data.DataLoader(
    jigsaw_test,
    batch_size=jigsaw_batch_size,
    shuffle=False,
    num_workers=jigsaw_workers,
    collate_fn=jigsaw_collator,
)
jigsaw_distill_train_loader = torch.utils.data.DataLoader(
    jigsaw_train, batch_size=jigsaw_batch_size, shuffle=True,
    num_workers=jigsaw_workers, collate_fn=jigsaw_distill_collator,
)
jigsaw_distill_val_loader = torch.utils.data.DataLoader(
    jigsaw_val, batch_size=jigsaw_batch_size, shuffle=False,
    num_workers=jigsaw_workers, collate_fn=jigsaw_distill_collator,
)
print("jigsaw data loaders initialized.")


## loss-functions,  optimizers and hyperparameters
loss_fn_teacher_mlm = None                          # model comes with its own loss function (?)
loss_fn_teacher = torch.nn.BCEWithLogitsLoss()      #?
loss_fn_student = torch.nn.BCEWithLogitsLoss()      #?

optimizer_teacher_mlm = torch.optim.AdamW(teacher_mlm.parameters(),lr=5e-5,)
optimizer_student = torch.optim.AdamW(student_model_1.parameters(),lr=5e-5,)

student_alpha = 0.5         #??
student_temperature = 2.0   #??


print("Setup done")



############################
#####     TRAINING     #####
############################

print("\n\n")
print("####################################################")
print("### PHASE 1 : DOMAIN ADAPTATION OF TEACHER MODEL ###")
print("####################################################")
print("\n\n")


print("Starting domain adaptation training for teacher model...")
epochs_da = 5
update_pr_da = 10
prints_da = 10
patience_da = 5
best_loss_da = float('inf')
best_epoch_da = 0
best_teacher_da_model_state_dict = copy.deepcopy(teacher_mlm.state_dict())

teacher_domain_train_losses = []
teacher_domain_val_losses = []
patience_counter_da = 0


for epoch in range(epochs_da):
    domain_adaption_results = modelFunctions.train_model_unsupervised(
        model = teacher_mlm,
        dataloader = jef1056_train_loader,
        optimizer = optimizer_teacher_mlm,
        device = device,
        prints = prints_da,
    )
    teacher_domain_train_loss = domain_adaption_results
    teacher_domain_train_losses.append(teacher_domain_train_loss)

    # Validate the model for current epoch
    teacher_domain_val = modelFunctions.validate_model_unsupervised(
        model = teacher_mlm,
        dataloader = jef1056_val_loader,
        device = device,
        prints = prints_da,
    )
    teacher_domain_val_losses.append(teacher_domain_val)

    # Print results for current epoch if wanted
    if prints_da >= 1:
        if epoch == 0 or epoch % update_pr_da == 0:
            print(f"Domain Adaption - Epoch {epoch}/{epochs_da}")
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


    if prints_da >= 1:
        print(f"Best DA teacher model saved at epoch {best_epoch_da} with val loss: {best_loss_da:.4f}")


    # Print results for current epoch if wanted
    if prints_da >= 1:
        if epoch % update_pr_da == 0:
            print(f"Domain Adaption - Epoch {epoch}/{epochs_da}")
            if prints_da >= 3:
                print(f"Domain Adaption - Train loss: {domain_adaption_results:.4f}")
                print(f"Domain Adaption - Val loss: {teacher_domain_val:.4f}")

torch.save(best_teacher_da_model_state_dict, f"{BEST_TEACHER_PATH}/{BEST_TEACHER_DA_NAME}")  # Save best model
print(f"Best DA teacher model saved at epoch {best_epoch_da} with val loss: {best_loss_da:.4f}, saved at path: {BEST_TEACHER_PATH}/{BEST_TEACHER_DA_NAME}")


print("Domain adaption done.")
   
teacher_mlm.save_pretrained("models/teacher_domain_adapted")
teacher_tokenizer.save_pretrained("models/teacher_domain_adapted")
print("Domain adapted Teacher model saved.")



print("\n\n")
print("####################################################")
print("### PHASE 2 : TRANSFER LEARNING OF TEACHER MODEL ###")
print("####################################################")
print("\n\n")

teacher_model = AutoModelForSequenceClassification.from_pretrained(
    "models/teacher_domain_adapted",
    num_labels=6,
    problem_type="multi_label_classification",
)

optimizer_teacher = torch.optim.AdamW(teacher_model.parameters(),lr=5e-5,)
print("Teacher model re-initialized for transfer learning.")

print("Starting transfer learning for teacher model...")
epochs_tl = 5
update_pr_tl = 10
patience_tl = 5
best_loss_tl = float('inf')
best_epoch_tl = 0
prints_tl = 10

teacher_transfer_train_losses = []
teacher_transfer_train_accuracies_total = []
teacher_transfer_train_accuracies_individual = []
teacher_transfer_val_losses = []
teacher_transfer_val_accuracies_total = []
teacher_transfer_val_accuracies_individual = []
patience_counter_tl = 0

for epoch in range(epochs_tl):
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

    # Print results for current epoch if wanted
    if prints_tl >= 1:
        if epoch % update_pr_tl == 0:
            print(f"Transfer Learning - Epoch {epoch}/{epochs_tl}")
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

    if prints_da >= 1 and epoch % update_pr_da == 0:
        print(f"Best DA epoch: {best_epoch_da}, validation loss: {best_loss_da:.4f}")


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
epochs_dis = 5
update_pr_dis = 10
patience_dis = 5
best_loss_dis = float('inf')
best_epoch_dis = 0
prints_dis = 10

student_distil_losses = []
student_distil_accuracies_total = []
student_distil_accuracies_individual = []
patience_counter_dis = 0
best_student_model_1_state_dict = copy.deepcopy(student_model_1.state_dict())

for epoch in range(epochs_dis):
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
    if prints_dis >= 1:
        if epoch % update_pr_dis == 0:
            print(f"Student Distilling - Epoch {epoch}/{epochs_dis}")
            if prints_dis >= 3:
                print(f"Student Distilling - Train loss: {student_train_loss:.4f}, Train accuracy (total): {student_train_accuracy_total:.4f}, Train accuracy (individual): {student_train_accuracy_individual:.4f}")
                print(f"Student Distilling - Val loss: {student_val_loss:.4f}, Val accuracy (total): {student_val_accuracy_total:.4f}, Val accuracy (individual): {student_val_accuracy_individual:.4f}")

torch.save(best_student_model_1_state_dict, f"{BEST_STUDENT_PATH}/best_student_model_1_state_dict.pth")
print(f"Best student model saved at epoch {best_epoch_dis} with val loss: {best_loss_dis:.4f}")


print("\n\n")
print("################################")
print("### PHASE 4 : TESTING MODELS ###")
print("################################")
print("\n\n")

## Test teacher model on labeled data
## Test student model(s) on labeled data





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




#if __name__ == "__main__":
#    main()