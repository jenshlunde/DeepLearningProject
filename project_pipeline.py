###########################
#####     IMPORTS     #####
###########################

import argparse
import copy
from pathlib import Path
from typing import Optional, Sequence
import torch
from transformers import (AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)
from torch.utils.data import random_split

import csv                                         

import jsonDataset                                                  #Homegrown dataset class for reading json files
import modelFunctions                                               #Homegrown functions for models: train, val, test, ...
import time                                                         #Timing functions to see how long it takes to train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-da", type=int, default=100)
    parser.add_argument("--epochs-tl", type=int, default=100)
    parser.add_argument("--epochs-distill", type=int, default=100)
    parser.add_argument("--batch-size-da", type=int, default=16)
    parser.add_argument("--batch-size-tl", type=int, default=16)
    parser.add_argument("--batch-size-dis", type=int, default=16)
    parser.add_argument("--num-workers-da", type=float, default=4)
    parser.add_argument("--num-workers-tl", type=float, default=4)
    parser.add_argument("--num-workers-dis", type=float, default=4)
    parser.add_argument("--max-length-da", type=int, default=512)
    parser.add_argument("--max-length-tl", type=int, default=512)
    parser.add_argument("--max-length-dis", type=int, default=512)
    parser.add_argument("--learning-rate-da", type=float, default=5e-5)
    parser.add_argument("--learning-rate-tl", type=float, default=5e-5)
    parser.add_argument("--learning-rate-dis", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--temperature", type=float, default=2.)
    parser.add_argument("--patience-da", type=int, default=5)
    parser.add_argument("--patience-tl", type=int, default=5)
    parser.add_argument("--patience-dis", type=int, default=5)
    parser.add_argument("--do-DA", action="store_true", default=False)
    parser.add_argument("--do-TL", action="store_true", default=False)
    parser.add_argument("--do-DIS", action="store_true", default=False)
    parser.add_argument("--do-STUDENT_TESTS", action="store_true", default=False)
    parser.add_argument("--test-run", action="store_true", default=False)
    return parser

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    
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

    #PATHS
    output_path = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/Models")
    output_path.mkdir(parents=True, exist_ok=True)
    BEST_DA_TEACHER_PATH = output_path / "teacher_DA"
    BEST_TL_TEACHER_PATH = output_path / "teacher_TL"
    BEST_STUDENTS_PATH = output_path / "best_students"
    DA_MODEL_PATH = BEST_DA_TEACHER_PATH / "MODEL"
    TL_MODEL_PATH = BEST_TL_TEACHER_PATH / "MODEL"
    CACHE_LOGITS_PATH = output_path / "logits_cache"

    if args.test_run:   
        print("### RUNNING TEST - EPOCHS AND DATALENGTHS REDUCED ###")
        #SHORT DATASETS FOR TESTS
        JEF_1056_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/discord-v3-antispam_short_100.jsonl")                                                    
        JIGSAW_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_train_short.jsonl")
        #OVERWRITE EPOCH LENGTHS AND DATASET LENGTHS FOR TESTING
        args.epochs_da = 3
        args.epochs_tl = 3
        args.epochs_dis = 3
        args.max_length_da = 32
        args.max_length_tl = 32
        args.max_length_dis = 32
        #TEST ALL PHASES
        args.do_DA = True
        args.do_TL = True
        args.do_DIS = True
        args.do_STUDENT_TESTS = True
    else:           
        JEF_1056_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/discord-v3-antispam_balanced_samples.jsonl")                                            
        JIGSAW_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_train.jsonl")        
    JIGSAW_DATASET_TEST_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_test.jsonl")
    
    ## Teacher model(s)
    TEACHER_BASE = "answerdotai/ModernBERT-base"
    teacher_tokenizer = None
    teacher_model = None
    jigsaw_data_set = None

    print("General Setup done")

    if args.do_DA:
        
        # MLM models automatically uses CrossEntropyLoss with ignore_index=-100 for masked tokens, so no need to specify loss function here.
        teacher_mlm = AutoModelForMaskedLM.from_pretrained(TEACHER_BASE)
        print(f"Teacher model initialized: {TEACHER_BASE}")
        teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE)
        print(f"Teacher tokenizer initialized: {TEACHER_BASE}")
    
        print("\n\n")
        print("####################################################")
        print("### PHASE 1 : DOMAIN ADAPTATION OF TEACHER MODEL ###")
        print("####################################################")
        print("\n\n")

        ### unlabeled dataloader ###
        data_loader_t0 = time.perf_counter()
        jef1056_dataset = jsonDataset.jef1056dataset(JEF_1056_DATASET_PATH)
        jef_train_size = int(0.9 * len(jef1056_dataset))
        jef_val_size = len(jef1056_dataset) - jef_train_size
        jef_train, jef_val = random_split(jef1056_dataset, [jef_train_size, jef_val_size],generator=torch.Generator().manual_seed(args.seed))
        
        print("jef train data set initialized, length:", len(jef_train))
        print("jef validation data set initialized, length:", len(jef_val))

        mlm_collator = jsonDataset.MlmCollator(teacher_tokenizer, max_length=args.max_length_da)

        data_loader_t1 = time.perf_counter()
        print(f"Data loading time: {data_loader_t1 - data_loader_t0:.2f} seconds")

        # Freeze some of the model to lighten work - embeddings and half of the encoder layers
        num_layers_DA = len(teacher_mlm.model.layers) // 2 
        modelFunctions.freeze_layers(teacher_mlm, freeze_embeddings=True, keep_last=num_layers_DA, verbose=True) 

        #Training setup - DA
        training_args_da = TrainingArguments(
            output_dir=BEST_DA_TEACHER_PATH,                    #Where model predictions and checkpoints will be written
            per_device_train_batch_size=args.batch_size_da,     #Default = 8
            num_train_epochs=args.epochs_da,                    #Default = 3
            learning_rate=args.learning_rate_da,                #Default = 5e-5
            dataloader_num_workers=args.num_workers_da,         #Default = 0
            dataloader_persistent_workers=True,                 #Default = False
            load_best_model_at_end=True,                        #Default = False
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
        )
        callback_args_DA = [EarlyStoppingCallback(early_stopping_patience=args.patience_da)]
        
        DA_Trainer = Trainer(model = teacher_mlm, train_dataset = jef_train, eval_dataset = jef_val,
                            data_collator = mlm_collator, callbacks = callback_args_DA, args = training_args_da
                            )

        print("DA Setup done - Starting domain adaptation training for teacher model... \n")
        trainer_timer_start = time.perf_counter()
        DA_Trainer.train()
        trainer_timer_end = time.perf_counter()
        print(f"Trainer time: {trainer_timer_end - trainer_timer_start:.2f} seconds")
        print("Domain adaption done.")

        DA_Trainer.save_model(DA_MODEL_PATH)
        teacher_tokenizer.save_pretrained(BEST_DA_TEACHER_PATH / "TOKENIZER")
        print("Domain adapted Teacher model saved.")

    if args.do_TL:
        print("\n\n")
        print("####################################################")
        print("### PHASE 2 : TRANSFER LEARNING OF TEACHER MODEL ###")
        print("####################################################")
        print("\n\n")

        ## labeled dataloader
        jigsaw_data_set = jsonDataset.jigsawDataset(JIGSAW_DATASET_PATH)
        jigsaw_train_size = int(0.9 * len(jigsaw_data_set))
        jigsaw_val_size = len(jigsaw_data_set) - jigsaw_train_size
        jigsaw_train, jigsaw_val = random_split(jigsaw_data_set, [jigsaw_train_size, jigsaw_val_size], generator=torch.Generator().manual_seed(args.seed))
        
        print("jigsaw_train data set initialized, length:", len(jigsaw_train))
        print("jigsaw_val data set initialized, length:", len(jigsaw_val))

        if teacher_tokenizer is None:
            teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE) #If not initilized in DA phase, initialize here

        jigsaw_collator = jsonDataset.SupervisedCollator(teacher_tokenizer,max_length=args.max_length_tl)
            
        print("Re-initializing teacher model for transfer learning.")
        # Models automatically uses BinaryCrossEntropy with logits loss when num_labels > 1 and problem_type="multi_label_classification", so no need to specify loss function here.
        teacher_model = AutoModelForSequenceClassification.from_pretrained(DA_MODEL_PATH, num_labels=6,problem_type="multi_label_classification")
        modelFunctions.freeze_layers(teacher_model, freeze_embeddings=True, keep_last=3, verbose=True)
        
        #Training setup - TL
        training_args_TL = TrainingArguments(
            output_dir=BEST_TL_TEACHER_PATH,                    #Where model predictions and checkpoints will be written
            per_device_train_batch_size=args.batch_size_tl,     #Default = 8
            num_train_epochs=args.epochs_tl,                    #Default = 3
            learning_rate=args.learning_rate_tl,                #Default = 5e-5
            dataloader_num_workers=args.num_workers_tl,         #Default = 0
            dataloader_persistent_workers=True,                 #Default = False
            load_best_model_at_end=True,                        #Default = False
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
        )
        callback_args_TL = [EarlyStoppingCallback(early_stopping_patience=args.patience_tl)]
        
        TL_Trainer = Trainer(model = teacher_model, train_dataset = jigsaw_train, eval_dataset = jigsaw_val,
                             data_collator = jigsaw_collator, callbacks = callback_args_TL, args = training_args_TL
                            )

        print("TL Setup done - Starting Transfer Learning training for teacher model... \n")
        TL_Trainer.train()
        print("Transfer Learning done.")
        
        TL_Trainer.save_model(TL_MODEL_PATH)
        teacher_tokenizer.save_pretrained(BEST_TL_TEACHER_PATH / "TOKENIZER")
        print("Transfer Learning Teacher model saved.")


    if args.do_DIS:
        print("\n\n")
        print("##########################################")
        print("### PHASE 3 : DISTILLING ONTO STUDENTS ###")
        print("##########################################")
        print("\n\n")

        #If not initilized in prior phases, initialize here
        if teacher_model is None:
            teacher_model = AutoModelForSequenceClassification.from_pretrained(TL_MODEL_PATH, num_labels=6,problem_type="multi_label_classification")
    
        #Freeze all of the model   
        num_layers_DIS = len(teacher_model.model.layers)
        modelFunctions.freeze_layers(teacher_model, freeze_embeddings=True, keep_last=num_layers_DIS, verbose=True)

        #REDUCE TOKENIZER???
        if teacher_tokenizer is None:
            teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE) 
        #student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE)                            ##?? should look more into tokenizers

        #if not intialized in prior phases, initialize here
        if jigsaw_data_set is None:
            jigsaw_data_set = jsonDataset.jigsawDataset(JIGSAW_DATASET_PATH)
            jigsaw_train_size = int(0.9 * len(jigsaw_data_set))
            jigsaw_val_size = len(jigsaw_data_set) - jigsaw_train_size
            jigsaw_train, jigsaw_val = random_split(jigsaw_data_set, [jigsaw_train_size, jigsaw_val_size], generator=torch.Generator().manual_seed(args.seed))


        jigsaw_collator = jsonDataset.SupervisedCollator(teacher_tokenizer,max_length=args.max_length_tl)
        train_cache = modelFunctions.cache_logits(teacher_model, jigsaw_train, jigsaw_collator, cache_path=CACHE_LOGITS_PATH / "train_logits_cache", batch_size=args.batch_size_dis, device=device, seed=args.seed)
        val_cache = modelFunctions.cache_logits(teacher_model, jigsaw_val, jigsaw_collator, cache_path=CACHE_LOGITS_PATH / "val_logits_cache", batch_size=args.batch_size_dis, device=device, seed=args.seed)
                

        ## Student model(s)
        STUDENT_BASE = "bert-base-uncased"
        student_model_1 = AutoModelForSequenceClassification.from_pretrained(STUDENT_BASE, num_labels=6, problem_type="multi_label_classification")
        student_model_2 = AutoModelForSequenceClassification.from_pretrained(STUDENT_BASE, num_labels=6, problem_type="multi_label_classification")

        student_models = [student_model_1, student_model_2]

        for idx, student_model in enumerate(student_models):

            BEST_STUDENT_PATH = BEST_STUDENTS_PATH / f"student_{idx+1}"
            BEST_STUDENT_PATH.mkdir(parents=True, exist_ok=True)

            training_args_DIS = TrainingArguments(
                output_dir=BEST_STUDENT_PATH,                       #Where model predictions and checkpoints will be written
                per_device_train_batch_size=args.batch_size_dis,     #Default = 8
                num_train_epochs=args.epochs_dis,                    #Default = 3
                learning_rate=args.learning_rate_dis,                #Default = 5e-5
                dataloader_num_workers=args.num_workers_dis,         #Default = 0
                dataloader_persistent_workers=True,                 #Default = False
                load_best_model_at_end=True,                        #Default = False
                eval_strategy="epoch",
                save_strategy="epoch",
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=2,
            )
            callback_args_DIS = [EarlyStoppingCallback(early_stopping_patience=args.patience_dis)]
            
            #DIS_Trainer = DistillationTrainer(model = student_model, train_dataset = train_cache, eval_dataset = val_cache,
            #                                  data_collator = DistillationCollator(student_tokenizer, args.max_length),
            #                                  callbacks = callback_args_DIS, args = training_args_DIS
            #                                  alpha = args.alpha, temperature = args.temperature
            #                                  )

            print(f"DIS Setup done - Starting Distillation student model {idx+1} \n")
            #DIS_Trainer.train()
            print("Distillation done.")
            
            #DIS_Trainer.save_model(BEST_STUDENT_PATH)
            print(f"Distillation Student model {idx+1} saved.")





        teacher_cache = 1 # ?


    if args.do_STUDENT_TESTS:
        print("\n\n")
        print("################################")
        print("### PHASE 4 : TESTING MODELS ###")
        print("################################")
        print("\n\n")

        jigsaw_data_set = jsonDataset.jigsawDataset(JIGSAW_DATASET_TEST_PATH)
            

        """#ONLY FOR TESTING LATER ? 
        
        #jigsaw_test = jsonDataset.jigsawDataset("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_test_short.jsonl")
        #print("jigsaw_test data set initialized, length:", len(jigsaw_test))

        jigsaw_test_loader = torch.utils.data.DataLoader(jigsaw_test, batch_size=jigsaw_batch_size, shuffle=False, 
                                                        num_workers=jigsaw_workers, persistent_workers=True, collate_fn=jigsaw_collator,
                                                        )

         
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
        """


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