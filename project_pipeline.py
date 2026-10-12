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
    parser.add_argument("--epochs-dis", type=int, default=100)
    parser.add_argument("--batch-size-da", type=int, default=8)             #DA usually has larger datasets, so smaller batch size to fit in memory
    parser.add_argument("--batch-size-tl", type=int, default=16)
    parser.add_argument("--batch-size-dis", type=int, default=64)           #Student models are smaller, so more memory available for larger batch size
    parser.add_argument("--num-workers-da", type=float, default=4)
    parser.add_argument("--num-workers-tl", type=float, default=4)
    parser.add_argument("--num-workers-dis", type=float, default=4)
    parser.add_argument("--max-length-da", type=int, default=256)
    parser.add_argument("--learning-rate-da", type=float, default=5e-5)
    parser.add_argument("--learning-rate-tl", type=float, default=1e-5)
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
    parser.add_argument("--do-MET", action="store_true", default=False)    
    parser.add_argument("--test-run", action="store_true", default=False)
    parser.add_argument("--full-run", action="store_true", default=False)
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
    TOKENIZER_TEACHER_PATH = BEST_TL_TEACHER_PATH / "TOKENIZER"
    TOKENIZER_STUDENT_PATH = BEST_STUDENTS_PATH / "TOKENIZER"

    if args.full_run:
        print("### RUNNING FULL TRAINING PIPELINE ###")
        args.do_DA = True
        args.do_TL = True
        args.do_DIS = True
        args.do_MET = True
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
        #TEST ALL PHASES
        args.do_DA = True
        args.do_TL = True
        args.do_DIS = True
        args.do_MET = True
    else:           
        JEF_1056_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/discord-v3-antispam_short_100.jsonl")                                                    
        #JEF_1056_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/discord-v3-antispam_balanced_samples.jsonl")                                            
        JIGSAW_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_train_short.jsonl")
        #JIGSAW_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_train.jsonl")        
    JIGSAW_DATASET_TEST_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_test_short.jsonl")
    
    ## Teacher model(s)
    TEACHER_BASE = "answerdotai/ModernBERT-base"
    STUDENT_BASE = "bert-base-uncased"
    teacher_model = None
    student_models = []
    teacher_tokenizer = None
    student_tokenizer = None
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

        # Freeze some of the model to lighten work - half of the encoder layers - dont freeze any embedding as that is what learns words
        num_layers_DA = len(teacher_mlm.model.layers) // 2 
        modelFunctions.freeze_layers(teacher_mlm, freeze_embeddings=False, keep_last=num_layers_DA, verbose=True) 

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

        if teacher_tokenizer is None:   #If not initilized in DA phase, initialize here
            if TOKENIZER_TEACHER_PATH.exists():
                teacher_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_TEACHER_PATH)   #If a saved version exists, load that one
            else:
                teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE)     #Otherwise, load the base version of the teacher model tokenizer

        jigsaw_collator = jsonDataset.SupervisedCollator(teacher_tokenizer)
            
        print("Re-initializing teacher model for transfer learning.")
        # Models automatically uses BinaryCrossEntropy with logits loss when num_labels > 1 and problem_type="multi_label_classification", so no need to specify loss function here.
        teacher_model = AutoModelForSequenceClassification.from_pretrained(DA_MODEL_PATH, num_labels=6,problem_type="multi_label_classification")
        modelFunctions.freeze_layers(teacher_model, freeze_embeddings=True, keep_last=1, verbose=True)
        
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
        modelFunctions.freeze_layers(teacher_model, verbose=True)

        #REDUCE TOKENIZER???
        if teacher_tokenizer is None:   #If not initilized in DA phase, initialize here
            if TOKENIZER_TEACHER_PATH.exists():
                teacher_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_TEACHER_PATH)   #If a saved version exists, load that one
            else:
                teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE)             #Otherwise, load the base version of the teacher model tokenizer
        
        #THIS SHOULD BE A SMALLER TOKENIZER THAN THE STUDENT_BASE?, BUT FOR NOW USE SAME AS TEACHER
        student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE)                            ##?? should look more into tokenizers

        #if not intialized in prior phases, initialize here
        if jigsaw_data_set is None:
            jigsaw_data_set = jsonDataset.jigsawDataset(JIGSAW_DATASET_PATH)
            jigsaw_train_size = int(0.9 * len(jigsaw_data_set))
            jigsaw_val_size = len(jigsaw_data_set) - jigsaw_train_size
            jigsaw_train, jigsaw_val = random_split(jigsaw_data_set, [jigsaw_train_size, jigsaw_val_size], generator=torch.Generator().manual_seed(args.seed))


        jigsaw_collator = jsonDataset.SupervisedCollator(teacher_tokenizer)
        jigsaw_train_cache = modelFunctions.cache_logits(teacher_model, jigsaw_train, jigsaw_collator, cache_path=CACHE_LOGITS_PATH / "train_logits_cache", batch_size=args.batch_size_dis, device=device, seed=args.seed)
        jigsaw_val_cache = modelFunctions.cache_logits(teacher_model, jigsaw_val, jigsaw_collator, cache_path=CACHE_LOGITS_PATH / "val_logits_cache", batch_size=args.batch_size_dis, device=device, seed=args.seed)
        print("Logits caching done for jigsaw dataset.")

        jef1056_dataset = jsonDataset.jef1056dataset(JEF_1056_DATASET_PATH)
        jef_train_size = int(0.9 * len(jef1056_dataset))
        jef_val_size = len(jef1056_dataset) - jef_train_size
        jef_train, jef_val = random_split(jef1056_dataset, [jef_train_size, jef_val_size], generator=torch.Generator().manual_seed(args.seed))
        jef_collator = jsonDataset.TextCollator(teacher_tokenizer, max_length=args.max_length_da)

        jef_train_cache = modelFunctions.cache_logits(teacher_model, jef_train, jef_collator,
                                                      cache_path=CACHE_LOGITS_PATH / "jef_train_logits_cache",
                                                      batch_size=args.batch_size_dis, device=device, seed=args.seed,
                                                      )
        jef_val_cache = modelFunctions.cache_logits(teacher_model, jef_val, jef_collator,
                                                    cache_path=CACHE_LOGITS_PATH / "jef_val_logits_cache",
                                                    batch_size=args.batch_size_dis, device=device, seed=args.seed,
                                                    )                
        print("Logits caching done for jef1056 dataset.")

        ### Create dual dataset for distillation - outside the loop to avoid reloading the caches for each student model and give each student the same data 
        dual_train = jsonDataset.DualDataset(jef_train_cache, jigsaw_train_cache)
        dual_val = jsonDataset.DualDataset(jef_val_cache, jigsaw_val_cache)
        print("Dual dataset for distillation created.")

        ## Student model(s)
        student_model_1 = AutoModelForSequenceClassification.from_pretrained(STUDENT_BASE, num_labels=6, problem_type="multi_label_classification")
        modelFunctions.freeze_layers(student_model_1, freeze_embeddings=True, keep_last=1, verbose=True) #To lighten during testing
        student_model_2 = AutoModelForSequenceClassification.from_pretrained(STUDENT_BASE, num_labels=6, problem_type="multi_label_classification")
        modelFunctions.freeze_layers(student_model_2, freeze_embeddings=True, keep_last=1, verbose=True) #To lighten during testing
        student_models = [student_model_1, student_model_2]
        print(f"{len(student_models)} student models initialized for distillation.")
            
        for idx, student_model in enumerate(student_models):
            BEST_STUDENT_PATH = BEST_STUDENTS_PATH / f"student_{idx+1}"
            BEST_STUDENT_PATH.mkdir(parents=True, exist_ok=True)

            training_args_DIS = TrainingArguments(
                output_dir=BEST_STUDENT_PATH,                       #Where model predictions and checkpoints will be written
                per_device_train_batch_size=args.batch_size_dis,    #Default = 8
                num_train_epochs=args.epochs_dis,                   #Default = 3
                learning_rate=args.learning_rate_dis,               #Default = 5e-5
                dataloader_num_workers=args.num_workers_dis,        #Default = 0
                dataloader_persistent_workers=True,                 #Default = False
                load_best_model_at_end=True,                        #Default = False
                eval_strategy="epoch",
                save_strategy="epoch",
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=2,
                remove_unused_columns=False,
                )
            callback_args_DIS = [EarlyStoppingCallback(early_stopping_patience=args.patience_dis)]
            
            DIS_Trainer = modelFunctions.DualDistillationTrainer(
                model=student_model,
                train_dataset=dual_train,
                eval_dataset=dual_val,
                data_collator=jsonDataset.DualDistillationCollator(student_tokenizer, args.max_length_da),
                callbacks=callback_args_DIS,
                args=training_args_DIS,
                alpha=args.alpha,
                temperature=args.temperature,
            )

            print(f"DIS Setup done - Starting Distillation student model {idx+1} \n")
            DIS_Trainer.train()
            print("Distillation done.")
            
            DIS_Trainer.save_model(BEST_STUDENT_PATH)
            print(f"Distillation Student model {idx+1} saved.")
    
        student_tokenizer.save_pretrained(TOKENIZER_STUDENT_PATH)
        print(f"Student tokenizer saved.")

    if args.do_MET:
        print("\n\n")
        print("################################")
        print("### PHASE 4 : TESTING MODELS ###")
        print("################################")
        print("\n\n")

        #Initialize teacher model and tokenizers if not already done in prior phases
        if teacher_model is None:
            if not TL_MODEL_PATH.exists():
                raise FileNotFoundError(f"Teacher checkpoint not found: {TL_MODEL_PATH}. Run earlier phases first.")
            teacher_model = AutoModelForSequenceClassification.from_pretrained(TL_MODEL_PATH, num_labels=6, problem_type="multi_label_classification")

        if teacher_tokenizer is None:   #If not initilized in DA phase, initialize here
            tokenizer_path = BEST_TL_TEACHER_PATH / "TOKENIZER"
            if tokenizer_path.exists():
                teacher_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)   #If a saved version exists, load that one
            else:
                teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE)     #Otherwise, load the base version of the teacher model tokenizer


        ### Load evaluation dataset                
        evaluation_dataset = jsonDataset.jigsawDataset(JIGSAW_DATASET_TEST_PATH)
        label_names = evaluation_dataset.class_names

        #EVAL TEACHER MODEL
        evaluation_teacher_collator = jsonDataset.SupervisedCollator(teacher_tokenizer) 
        evaluation_teacher_args = TrainingArguments(
            output_dir=BEST_TL_TEACHER_PATH / "evaluation",
            per_device_eval_batch_size=args.batch_size_tl,
            dataloader_num_workers=int(args.num_workers_tl),
            remove_unused_columns=False,
            report_to=[],
        )
        teacher_evaluator = Trainer(
            model=teacher_model,
            args=evaluation_teacher_args,
            eval_dataset=evaluation_dataset,
            data_collator=evaluation_teacher_collator,
            compute_metrics=modelFunctions.compute_multilabel_metrics,
        )
        teacher_metrics = teacher_evaluator.evaluate(metric_key_prefix="teacher")
        teacher_benchmark = modelFunctions.evaluate_multilabel_model(
            teacher_model,
            evaluation_dataset,
            evaluation_teacher_collator,
            device=device,
            batch_size=args.batch_size_tl,
            num_workers=int(args.num_workers_tl),
            label_names=label_names,
            model_path=TL_MODEL_PATH,
        )
        print("\nTeacher metrics:")
        print(teacher_metrics)
        print("Teacher end-to-end benchmark:")
        print(teacher_benchmark["benchmark"])



        #EVAL STUDENT MODELS
        if student_tokenizer is None:   #If not initilized in DA phase, initialize here
            if TOKENIZER_STUDENT_PATH.exists():
                student_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_STUDENT_PATH)   #If a saved version exists, load that one
            else:
                student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE)     #Otherwise, load the base version of the teacher model tokenizer

        student_collator = jsonDataset.SupervisedCollator(student_tokenizer)

        #Get all student checkpoint paths
        student_checkpoint_paths = sorted(path for path in BEST_STUDENTS_PATH.glob("student_*") if path.is_dir()) 

        # Evaluate each student model
        for student_index in range(len(student_checkpoint_paths)):
            
            student_path = student_checkpoint_paths[student_index]      #Path for current student          
            if student_index < len(student_models):                     #If student is already loaded use them
                student_model = student_models[student_index]
            else:                                                       #Else load  
                student_model = AutoModelForSequenceClassification.from_pretrained(student_path, num_labels=6, problem_type="multi_label_classification")

            student_evaluation_args = TrainingArguments(        
                output_dir=student_path / "evaluation",
                per_device_eval_batch_size=args.batch_size_dis,
                dataloader_num_workers=int(args.num_workers_dis),
                remove_unused_columns=False,
                report_to=[],
            )
            student_evaluator = Trainer(
                model=student_model,
                args=student_evaluation_args,
                eval_dataset=evaluation_dataset,
                data_collator=student_collator,
                compute_metrics=modelFunctions.compute_multilabel_metrics,
            )

            student_metrics = student_evaluator.evaluate(metric_key_prefix=f"student_{student_index + 1}")
            
            student_benchmark = modelFunctions.evaluate_multilabel_model(
                student_model,
                evaluation_dataset,
                student_collator,
                device=device,
                batch_size=args.batch_size_dis,
                num_workers=int(args.num_workers_dis),
                label_names=label_names,
                model_path=student_path,
            )
            print(f"\nStudent {student_index + 1} metrics:")
            print(student_metrics)
            print(f"Student {student_index + 1} end-to-end benchmark:")
            print(student_benchmark["benchmark"])


#### TODO: 

# Save results from training, validation and testing for all models in a structured way (e.g., CSV, JSON, or a database) for later analysis and comparison.
# 
# DO MORE STUDENT MODELS, DIFFERENT ARCHITECTURES, HYPERPARAMETERS, ETC.
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