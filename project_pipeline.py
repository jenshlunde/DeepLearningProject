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
from safetensors.torch import load_file
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import csv                                         
import json
import os
import time                                                         

import jsonDataset                                                  #Homegrown dataset class for reading json files
import modelFunctions                                               #Homegrown functions for models: train, val, test, ...
import modelSetups                                                  #Homegrown model setups: MLP, CNN, RNN, ...

# Options/variables to use for the training pipeline:
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-da", type=int, default=100)
    parser.add_argument("--epochs-tl", type=int, default=100)
    parser.add_argument("--epochs-dis", type=int, default=100)
    parser.add_argument("--batch-size-da", type=int, default=8)             #DA usually has larger datasets, so smaller batch size to fit in memory
    parser.add_argument("--batch-size-tl", type=int, default=16)
    parser.add_argument("--batch-size-dis", type=int, default=64)           #Student models are smaller, so more memory available for larger batch size
    parser.add_argument("--num-workers-da", type=int, default=4)
    parser.add_argument("--num-workers-tl", type=int, default=4)
    parser.add_argument("--num-workers-dis", type=int, default=4)
    parser.add_argument("--max-length-da", type=int, default=256)
    parser.add_argument("--max-length-tl", type=int, default=512)
    parser.add_argument("--max-length-dis", type=int, default=512)
    parser.add_argument("--learning-rate-da", type=float, default=5e-5)
    parser.add_argument("--learning-rate-tl", type=float, default=1e-5)
    parser.add_argument("--learning-rate-dis", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--temperature", type=float, default=2.)
    parser.add_argument("--patience-da", type=int, default=5)
    parser.add_argument("--patience-tl", type=int, default=5)
    parser.add_argument("--patience-dis", type=int, default=5)
    parser.add_argument("--do-DA", action="store_true", default=False)
    parser.add_argument("--do-TL", action="store_true", default=False)
    parser.add_argument("--do-TL-no-DA", action="store_true", default=False)
    parser.add_argument("--do-DIS", action="store_true", default=False)
    parser.add_argument("--do-MET", action="store_true", default=False)    
    parser.add_argument("--test-run", action="store_true", default=False)
    parser.add_argument("--full-run", action="store_true", default=False)
    parser.add_argument("--on-cluster", action="store_true", default=False)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1,
                        help=argparse.SUPPRESS)
    return parser

# Project pipeline main function
def main(argv: Optional[Sequence[str]] = None) -> None:
    # Read previous defined options
    args = build_parser().parse_args(argv)
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    
    if local_rank == 0:
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

    # if cuda is available - use. Otherwise use cpu
    if torch.cuda.is_available():
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
        cuda_index = torch.cuda.current_device()
        print("GPU available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name(cuda_index))
        device = torch.device("cuda", cuda_index)
        cuda = device
        print(torch.version.cuda)
        print("Current CUDA device:", cuda_index)
        print(torch.cuda.device_count())
        t = torch.cuda.get_device_properties(cuda_index).total_memory
        print(f"Total memory: {t/1e9} GB")
        r = torch.cuda.memory_reserved(cuda_index)
        a = torch.cuda.memory_allocated(cuda_index)
        f = r-a  # free inside reserved
        print(f"Reserved memory: {r/1e9} GB")
    else:
        device = torch.device('cpu')
        print("no cuda available")

    if local_rank == 0:
        print("Checks done")

    #########################
    #####     SETUP     #####
    #########################

    #PATHS
    if args.on_cluster: # if we are on the cluster, use cluster path
        output_path = Path("/home/dl08e26/OUTPUT")
    else:
        output_path = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/Models") # if not on cluster, use local path
    output_path.mkdir(parents=True, exist_ok=True)
    # Suffixes to differentiate paths for different phases of the pipeline
    BEST_DA_TEACHER_PATH = output_path / "teacher_DA"
    BEST_TL_TEACHER_PATH = output_path / "teacher_TL"
    BEST_STUDENTS_PATH = output_path / "best_students"
    DA_MODEL_PATH = BEST_DA_TEACHER_PATH / "MODEL"
    TL_MODEL_PATH = BEST_TL_TEACHER_PATH / "MODEL"
    CACHE_LOGITS_PATH = output_path / "logits_cache"
    TOKENIZER_PATH = output_path / "tokenizer"
    TOKENIZER_TEACHER_PATH = TOKENIZER_PATH / "TEACHER"
    TOKENIZER_STUDENT_PATH = TOKENIZER_PATH / "STUDENT"
    STUDENT_NAMES_PATH = output_path / "student_names.json"

    STUDENT_VOCAB_SIZE = 16_000     #Correct size??

    # If commanding --full-run, it will set all phases to true and therefore run all
    if args.full_run:
        if local_rank == 0:
            print("### RUNNING FULL TRAINING PIPELINE ###")
        args.do_DA = True
        args.do_TL = True
        args.do_DIS = True
        args.do_MET = True

    # If commanding --test-run, it will set all phases to true and reduce epochs and data lengths for testing purposes
    if args.test_run:   
        if local_rank == 0:
            print("### RUNNING TEST - EPOCHS AND DATALENGTHS REDUCED ###")
        #SHORT DATASETS FOR TESTS
        if args.on_cluster:
            JEF_1056_DATASET_PATH = Path("/home/dl08e26/DATA/discord-v3-antispam_short_100.jsonl")
            JIGSAW_DATASET_PATH = Path("/home/dl08e26/DATA/jigsaw_data_train_short.jsonl")
            JIGSAW_DATASET_TEST_PATH = Path("/home/dl08e26/DATA/jigsaw_data_test_short.jsonl")
        else:
            JEF_1056_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/discord-v3-antispam_short_100.jsonl")
            JIGSAW_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_train_short.jsonl")
            JIGSAW_DATASET_TEST_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_test_short.jsonl")
        #OVERWRITE EPOCH LENGTHS AND DATASET LENGTHS FOR TESTING
        args.epochs_da = 3
        args.epochs_tl = 3
        args.epochs_dis = 3
        args.max_length_da = 32
        #TEST ALL PHASES
        args.do_DA = True
        args.do_TL = True
        args.do_TL_no_DA = True
        args.do_DIS = True
        args.do_MET = True
    else: # If not a test run, use the full datasets - meaning if not fullrun or testrun, use the full datasets for the phase that are selected
        if args.on_cluster:
            JEF_1056_DATASET_PATH = Path("/home/dl08e26/DATA/discord-v3-antispam_balanced_samples.jsonl")
            JIGSAW_DATASET_PATH = Path("/home/dl08e26/DATA/jigsaw_data_train.jsonl")
            JIGSAW_DATASET_TEST_PATH = Path("/home/dl08e26/DATA/jigsaw_data_test.jsonl")
        else:
            JEF_1056_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/discord-v3-antispam_balanced_samples.jsonl")
            JIGSAW_DATASET_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_train.jsonl")
            JIGSAW_DATASET_TEST_PATH = Path("C:/Users/jensh/Desktop/Kaggle_disord_data_v3/jigsaw-toxic-comment-classification-challenge/versions/1/jigsaw_data_test.jsonl")

    ## Teacher model(s)
    TEACHER_BASE = "answerdotai/ModernBERT-base" # this is the nice modern model we want to use for the teacher
    STUDENT_BASE = "bert-base-uncased" # this is the sligthly shitter model we want to use for the student
    # variables to be filled in later
    teacher_model = None
    student_models = []
    teacher_tokenizer = None
    student_tokenizer = None
    jigsaw_data_set = None

    if local_rank == 0:
        print("General Setup done")

    #If command is --do-DA, run the domain adaptation phase for the teacher model
    if args.do_DA:
        if local_rank == 0:
            print("\n\n")
            print("####################################################")
            print("### PHASE 1 : DOMAIN ADAPTATION OF TEACHER MODEL ###")
            print("####################################################")
            print("\n\n")

        # MLM models automatically uses CrossEntropyLoss with ignore_index=-100 for masked tokens, so no need to specify loss function here.
        # (instantiate teacher model as as MLM and create the tokenizer)
        teacher_mlm = AutoModelForMaskedLM.from_pretrained(TEACHER_BASE) # Load the teacher model for masked language modeling (MLM) from the specified base model
        teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE) # Load the tokenizer for the teacher model from the specified base model
        if local_rank == 0:
            print(f"Teacher model initialized: {TEACHER_BASE}")
            print(f"Teacher tokenizer initialized: {TEACHER_BASE}")
    
        ### unlabeled dataloader ###
        data_loader_t0 = time.perf_counter() # Start timer for data loading - to see how long it takes
        jef1056_dataset = jsonDataset.jef1056dataset(JEF_1056_DATASET_PATH) # load discord dataset
        jef_train_size = int(0.9 * len(jef1056_dataset)) # Split the dataset into training and validation sets (90% for training, 10% for validation)
        jef_val_size = len(jef1056_dataset) - jef_train_size # Calculate the size of the validation set
        jef_train, jef_val = random_split(jef1056_dataset, [jef_train_size, jef_val_size],generator=torch.Generator().manual_seed(args.seed)) # Split the dataset into training and validation sets using a random split with a fixed seed for reproducibility

        if local_rank == 0:
            print("jef train data set initialized, length:", len(jef_train))
            print("jef validation data set initialized, length:", len(jef_val))

        mlm_collator = jsonDataset.MlmCollator(teacher_tokenizer, max_length=args.max_length_da) # make a data collator for MLM and tokenizer

        data_loader_t1 = time.perf_counter() # stop the time and prinout how long it took 
        if local_rank == 0:
            print(f"Data loading time: {data_loader_t1 - data_loader_t0:.2f} seconds")

        # Freeze some of the model to lighten work - half of the encoder layers - dont freeze any embedding as that is what learns words
        num_layers_DA = len(teacher_mlm.model.layers) // 2 # frezze half
        modelFunctions.freeze_layers(teacher_mlm, freeze_embeddings=False, keep_last=num_layers_DA, verbose=True) # actually freeze it

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
            ddp_find_unused_parameters=False,
        )
        callback_args_DA = [EarlyStoppingCallback(early_stopping_patience=args.patience_da)] # predefined early stopping number of epochs without improvement before stopping training
        
        DA_Trainer = Trainer(model = teacher_mlm, train_dataset = jef_train, eval_dataset = jef_val,
                            data_collator = mlm_collator, callbacks = callback_args_DA, args = training_args_da
                            ) # create the trainer object for domain adaptation training

        if local_rank == 0:
            print("DA Setup done - Starting domain adaptation training for teacher model... \n")

        # take time on how long it takes to train
        trainer_timer_start = time.perf_counter()
        DA_Trainer.train() # inbuilt train function of the trainer object to train the model - actually train it
        trainer_timer_end = time.perf_counter()

        # print stuff
        if local_rank == 0:
            print(f"Trainer time: {trainer_timer_end - trainer_timer_start:.2f} seconds")
            print("Domain adaption done.")
            DA_Trainer.save_model(DA_MODEL_PATH)
            teacher_tokenizer.save_pretrained(BEST_DA_TEACHER_PATH / "TOKENIZER")
            print("Domain adapted Teacher model saved.")

    # next phase 
    if args.do_TL:
        print("\n\n")
        print("####################################################")
        print("### PHASE 2 : TRANSFER LEARNING OF TEACHER MODEL ###")
        print("####################################################")
        print("\n\n")

        ## labeled dataloader
        jigsaw_data_set = jsonDataset.jigsawDataset(JIGSAW_DATASET_PATH) # load the jigsaw dataset for transfer learning
        # Split the dataset into training and validation sets (90% for training, 10% for validation)
        jigsaw_train_size = int(0.9 * len(jigsaw_data_set)) # train size
        jigsaw_val_size = len(jigsaw_data_set) - jigsaw_train_size # validation set size
        jigsaw_train, jigsaw_val = random_split(jigsaw_data_set, [jigsaw_train_size, jigsaw_val_size], generator=torch.Generator().manual_seed(args.seed)) # split into train and validation with correct size
        
        if local_rank == 0:
            print("jigsaw_train data set initialized, length:", len(jigsaw_train))
            print("jigsaw_val data set initialized, length:", len(jigsaw_val))

        # transfer learning without domain adaption
        if args.do_TL_no_DA:
            print("Transfer learning without domain adaptation selected.")
            teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_BASE, num_labels=6,problem_type="multi_label_classification")
            teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE)
        else:
            if teacher_tokenizer is None:   #If not initilized in DA phase, initialize here
                if TOKENIZER_TEACHER_PATH.exists():
                    teacher_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_TEACHER_PATH)  # should this not be teacher_best_path?? #If a saved version exists, load that one
                else:
                    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE)     #Otherwise, load the base version of the teacher model tokenizer
        
            print("Re-initializing teacher model for transfer learning.")
            # Models automatically uses BinaryCrossEntropy with logits loss when num_labels > 1 and problem_type="multi_label_classification", so no need to specify loss function here.
            teacher_model = AutoModelForSequenceClassification.from_pretrained(DA_MODEL_PATH, num_labels=6,problem_type="multi_label_classification") # Load the teacher model for sequence classification from the saved domain adapted model path with 6 labels and multi-label classification problem type

        jigsaw_collator = jsonDataset.SupervisedCollator(teacher_tokenizer, max_length=args.max_length_tl) # make a data collator for supervised learning and tokenizer
            
        num_layers_TL = len(teacher_model.model.layers) // 2 # freeze half of the teacher models layers
        modelFunctions.freeze_layers(teacher_model, freeze_embeddings=True, keep_last=num_layers_TL, verbose=True)
        
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
            ddp_find_unused_parameters=False,
        )
        callback_args_TL = [EarlyStoppingCallback(early_stopping_patience=args.patience_tl)] # default early stopping number
        
        TL_Trainer = Trainer(model = teacher_model, train_dataset = jigsaw_train, eval_dataset = jigsaw_val,
                             data_collator = jigsaw_collator, callbacks = callback_args_TL, args = training_args_TL
                            ) # create the trainer object for transfer learning training

        if local_rank == 0:
            print("TL Setup done - Starting Transfer Learning training for teacher model... \n")
        
        TL_Trainer.train() # actually train the combined model
        
        if local_rank == 0:
            print("Transfer Learning done.")
            TL_Trainer.save_model(TL_MODEL_PATH)
            teacher_tokenizer.save_pretrained(BEST_TL_TEACHER_PATH / "TOKENIZER")
            print("Transfer Learning Teacher model saved.")


    if args.do_DIS: # run distillation phase if command is --do-DIS

        print("\n\n")
        print("##########################################")
        print("### PHASE 3 : DISTILLING ONTO STUDENTS ###")
        print("##########################################")
        print("\n\n")

        if teacher_tokenizer is None:   #If not initilized in DA phase, initialize here
            if TOKENIZER_TEACHER_PATH.exists():
                teacher_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_TEACHER_PATH)   #If a saved version exists, load that one
            else:
                teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE)             #Otherwise, load the base version of the teacher model tokenizer

        #If not initilized in prior phases, initialize here
        if teacher_model is None:
            teacher_model = AutoModelForSequenceClassification.from_pretrained(TL_MODEL_PATH, num_labels=6,problem_type="multi_label_classification") # load the best transfer learning model
        #Freeze all of the model   
        modelFunctions.freeze_layers(teacher_model, verbose=True)

        #if not intialized in prior phases, initialize here
        if jigsaw_data_set is None:
            jigsaw_data_set = jsonDataset.jigsawDataset(JIGSAW_DATASET_PATH) # load
            jigsaw_train_size = int(0.9 * len(jigsaw_data_set)) # make train size
            jigsaw_val_size = len(jigsaw_data_set) - jigsaw_train_size # make val size
            jigsaw_train, jigsaw_val = random_split(jigsaw_data_set, [jigsaw_train_size, jigsaw_val_size], generator=torch.Generator().manual_seed(args.seed)) # split into sizes

        # make collator for supervised learning (train and validation)
        jigsaw_collator = jsonDataset.SupervisedCollator(teacher_tokenizer, max_length=args.max_length_dis)
        CACHE_BATCH = 32;
        # cache the logits/predictions for the jigsaw dataset to avoid recomputing them for each student model
        jigsaw_train_cache = modelFunctions.cache_logits(teacher_model, jigsaw_train, jigsaw_collator, cache_path=CACHE_LOGITS_PATH / "train_logits_cache", batch_size=CACHE_BATCH, device=device, seed=args.seed)
        jigsaw_val_cache   = modelFunctions.cache_logits(teacher_model, jigsaw_val,   jigsaw_collator, cache_path=CACHE_LOGITS_PATH / "val_logits_cache",   batch_size=CACHE_BATCH, device=device, seed=args.seed)
        print("Logits caching done for jigsaw dataset.")

        # same as above, just another dataset (discord)
        jef1056_dataset = jsonDataset.jef1056dataset(JEF_1056_DATASET_PATH) 
        jef_train_size = int(0.9 * len(jef1056_dataset))
        jef_val_size = len(jef1056_dataset) - jef_train_size
        jef_train, jef_val = random_split(jef1056_dataset, [jef_train_size, jef_val_size], generator=torch.Generator().manual_seed(args.seed))
        jef_collator = jsonDataset.TextCollator(teacher_tokenizer, max_length=args.max_length_da)

        jef_train_cache = modelFunctions.cache_logits(teacher_model, jef_train, jef_collator, cache_path=CACHE_LOGITS_PATH / "jef_train_logits_cache", batch_size=CACHE_BATCH, device=device, seed=args.seed)
        jef_val_cache   = modelFunctions.cache_logits(teacher_model, jef_val,   jef_collator, cache_path=CACHE_LOGITS_PATH / "jef_val_logits_cache",   batch_size=CACHE_BATCH, device=device, seed=args.seed)                
        print("Logits caching done for jef1056 dataset.")
        
        teacher_model.cpu()         #Empty the GPU memory for student models
        torch.cuda.empty_cache()

        # combine the two datasets
        ### Create dual dataset for distillation - outside the loop to avoid reloading the caches for each student model and give each student the same data 
        dual_train = jsonDataset.DualDataset(jef_train_cache, jigsaw_train_cache) # 
        dual_val = jsonDataset.DualDataset(jef_val_cache, jigsaw_val_cache)
        print("Dual dataset for distillation created.")

        #Create custom tokenizser
        if TOKENIZER_STUDENT_PATH.exists():                            
            student_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_STUDENT_PATH) # tokenize
        else:
            base = AutoTokenizer.from_pretrained(STUDENT_BASE)           # reuse BERT's pipeline
            student_tokenizer = base.train_new_from_iterator(modelFunctions.student_text_iterator(jef_train, jigsaw_train), vocab_size=STUDENT_VOCAB_SIZE)
            student_tokenizer.model_max_length = args.max_length_da      
            student_tokenizer.save_pretrained(TOKENIZER_STUDENT_PATH)    

        #Sanity check for tokenizer
        token_size = len(student_tokenizer)
        print(f"Student vocab: {token_size}")
        sample = [t for t, _ in itertools.islice(iter(jigsaw_train), 1000)]
        avg = lambda tok: sum(len(x) for x in tok(sample)["input_ids"]) / len(sample)
        print(f"avg tokens per Jigsaw comment: teacher {avg(teacher_tokenizer):.1f}, student {avg(student_tokenizer):.1f}")


        ### Student model(s) ###
        embed_dim_L = 256   # Set the embedding dimension (width of the tokens learned representation) for student models
        embed_dim_M = 64    # should be derived from something specific
        embed_dim_S = 16    # but for now this works

        token_size = len(student_tokenizer) # number of tokens in the student tokenizer

        #   _L (Large) ~10% of teacher model parameters
        #   _M (Medium) ~1% of teacher model parameters
        #   _S (Small) ~0.1% of teacher model parameters

        #MLP students 
        if args.test_run: # if test run, only use small student models to save time
            MLP_DeepNarrow_S = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_S, num_labels=6, layers_array=[128, 128, 128, 128], name="MLP_DeepNarrow_S")
            MLP_ShortWide_S = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_S, num_labels=6, layers_array=[512], name="MLP_ShortWide_S")
            MLP_Bottleneck_S = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_S, num_labels=6, layers_array=[256, 32, 256], name="MLP_Bottleneck_S")
            Trans_DeepNarrow_S = modelSetups.customTransformer(vocab_size=token_size, embed_dim=embed_dim_S, num_layers=4, d_model=32, dff=64, num_heads=2, num_labels=6, name="Trans_DeepNarrow_S")          #Odd number of num_heads gives error??
            Trans_ShortWide_S = modelSetups.customTransformer(vocab_size=token_size, embed_dim=embed_dim_S, num_layers=1, d_model=64, dff=128, num_heads=2, num_labels=6, name="Trans_ShortWide_S")           #Odd number of num_heads gives error??

            student_models = [MLP_DeepNarrow_S, MLP_ShortWide_S, MLP_Bottleneck_S, Trans_DeepNarrow_S, Trans_ShortWide_S]
        else: # otherwise run many student models to see which one performs best
            # deepnarrow have more but narrower layers // FULLY CONNECTED NETWORKS
            MLP_DeepNarrow_L = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_L, num_labels=6, layers_array=[512, 512, 512, 512, 512, 512, 512, 512], name="MLP_DeepNarrow_L")
            MLP_DeepNarrow_M = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_M, num_labels=6, layers_array=[256, 256, 256, 256, 256, 256], name="MLP_DeepNarrow_M")
            MLP_DeepNarrow_S = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_S, num_labels=6, layers_array=[128, 128, 128, 128], name="MLP_DeepNarrow_S")
            # shortwide have less but wider layers
            MLP_ShortWide_L = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_L, num_labels=6, layers_array=[4096, 4096], name="MLP_ShortWide_L")
            MLP_ShortWide_M = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_M, num_labels=6, layers_array=[1024, 1024], name="MLP_ShortWide_M")
            MLP_ShortWide_S = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_S, num_labels=6, layers_array=[512], name="MLP_ShortWide_S")
            # bottleneck have a narrow layer in the middle of the network
            MLP_Bottleneck_L = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_L, num_labels=6, layers_array=[2048, 512, 2048], name="MLP_Bottleneck_L")
            MLP_Bottleneck_M = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_M, num_labels=6, layers_array=[512, 64, 512], name="MLP_Bottleneck_M")
            MLP_Bottleneck_S = modelSetups.customMLP_wEmbedding(vocab_size=token_size, embed_dim=embed_dim_S, num_labels=6, layers_array=[256, 32, 256], name="MLP_Bottleneck_S")

            #Transformer students // Recurrent neural network
            Trans_DeepNarrow_L = modelSetups.customTransformer(vocab_size=token_size, embed_dim=embed_dim_L, num_layers=12, d_model=256, dff=512, num_heads=4, num_labels=6, name="Trans_DeepNarrow_L")      
            Trans_DeepNarrow_M = modelSetups.customTransformer(vocab_size=token_size, embed_dim=embed_dim_M, num_layers=6, d_model=128, dff=256, num_heads=2, num_labels=6, name="Trans_DeepNarrow_M")
            Trans_DeepNarrow_S = modelSetups.customTransformer(vocab_size=token_size, embed_dim=embed_dim_S, num_layers=4, d_model=32, dff=64, num_heads=2, num_labels=6, name="Trans_DeepNarrow_S")          #Odd number of num_heads gives error??
            Trans_ShortWide_L = modelSetups.customTransformer(vocab_size=token_size, embed_dim=embed_dim_L, num_layers=2, d_model=1024, dff=2048, num_heads=8, num_labels=6, name="Trans_ShortWide_L")
            Trans_ShortWide_M = modelSetups.customTransformer(vocab_size=token_size, embed_dim=embed_dim_M, num_layers=1, d_model=256, dff=512, num_heads=4, num_labels=6, name="Trans_ShortWide_M")
            Trans_ShortWide_S = modelSetups.customTransformer(vocab_size=token_size, embed_dim=embed_dim_S, num_layers=1, d_model=64, dff=128, num_heads=2, num_labels=6, name="Trans_ShortWide_S")           #Odd number of num_heads gives error??

            # dimension
            d_models_L = [256, 64, 1024]
            d_models_M = [128, 32, 128]
            d_models_S = [32, 8, 32]
            dffs_L = [512, 128, 512]
            dffs_M = [256, 64, 256]
            dffs_S = [64, 16, 64]
            num_heads_L = [4, 4, 4]
            num_heads_M = [2, 2, 2]
            num_heads_S = [2, 2, 2]
            Trans_Bottleneck_L = modelSetups.customTransformerBN(vocab_size=token_size, embed_dim=embed_dim_L, d_model=d_models_L, dff=dffs_L, num_heads=num_heads_L, num_labels=6, name="Trans_Bottleneck_L")
            Trans_Bottleneck_M = modelSetups.customTransformerBN(vocab_size=token_size, embed_dim=embed_dim_M, d_model=d_models_M, dff=dffs_M, num_heads=num_heads_M, num_labels=6, name="Trans_Bottleneck_M")
            Trans_Bottleneck_S = modelSetups.customTransformerBN(vocab_size=token_size, embed_dim=embed_dim_S, d_model=d_models_S, dff=dffs_S, num_heads=num_heads_S, num_labels=6, name="Trans_Bottleneck_S")

            student_models = [Trans_DeepNarrow_L, Trans_ShortWide_L, Trans_Bottleneck_L,
                              Trans_DeepNarrow_M, Trans_ShortWide_M, Trans_Bottleneck_M,
                              Trans_DeepNarrow_S, Trans_ShortWide_S, Trans_Bottleneck_S,
                              MLP_DeepNarrow_L,   MLP_ShortWide_L,      MLP_Bottleneck_L,
                              MLP_DeepNarrow_M,   MLP_ShortWide_M,      MLP_Bottleneck_M,
                              MLP_DeepNarrow_S,   MLP_ShortWide_S,      MLP_Bottleneck_S]

        
        student_models_names = [model.name for model in student_models]
        student_models_paths = [BEST_STUDENTS_PATH / model.name for model in student_models] # fill up paths for each student model to save them in the correct folder

        with open(STUDENT_NAMES_PATH, 'w') as f:    #Save student model names and paths to a JSON file for reference
            json.dump({"names": student_models_names, "paths": [str(p) for p in student_models_paths]}, f)

        print(f"{len(student_models)} student models initialized for distillation")

        #for each model/student, run the distillation process
        for idx, student_model in enumerate(student_models):
            model_name = student_model.name
            CURRENT_BEST_STUDENT_PATH = student_models_paths[idx]
            CURRENT_BEST_STUDENT_PATH.mkdir(parents=True, exist_ok=True) # make the path for the current student model if it does not exist yet

            print(f"\n\nStarting distillation for student model {idx+1}: {model_name} \n")                      #print which student model is being distilled
            print(f"Student model {idx+1} parameters: {sum(p.numel() for p in student_model.parameters())}")    #short print of number of parameters in student model

            # training values
            training_args_DIS = TrainingArguments(
                output_dir=CURRENT_BEST_STUDENT_PATH,               #Where model predictions and checkpoints will be written
                per_device_train_batch_size=args.batch_size_dis,    #Default = 8
                num_train_epochs=args.epochs_dis,                   #Default = 3
                learning_rate=args.learning_rate_dis,               #Default = 5e-5
                warmup_steps=1000,                                  # LR ramps linearly over first 1000 steps
                dataloader_num_workers=args.num_workers_dis,        #Default = 0
                dataloader_persistent_workers=True,                 #Default = False
                load_best_model_at_end=True,                        #Default = False
                eval_strategy="epoch",
                save_strategy="epoch",
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=2,
                remove_unused_columns=False,
                label_names=["labels"],
                )
            callback_args_DIS = [EarlyStoppingCallback(early_stopping_patience=args.patience_dis)] # predefined early stopping number of epochs without improvement before stopping training
            
            DIS_Trainer = modelFunctions.DualDistillationTrainer(
                model=student_model,
                train_dataset=dual_train,
                eval_dataset=dual_val,
                data_collator=jsonDataset.DualDistillationCollator(student_tokenizer, args.max_length_da),
                callbacks=callback_args_DIS,
                args=training_args_DIS,
                alpha=args.alpha,
                temperature=args.temperature,
            ) # actually create the trainer object for distillation training

            print(f"Setup done - Starting Distillation for student model {idx+1} \n")
            DIS_Trainer.train()
            print(f"Distillation for student model {idx+1} done.")
            
            DIS_Trainer.save_model(CURRENT_BEST_STUDENT_PATH)
            print(f"Distillation Student model {idx+1} saved.")
            student_model.cpu()                                     #Move student model away from GPU to free GPU mem
    
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
            print(f"Loading teacher model from {TL_MODEL_PATH}")
            teacher_model = AutoModelForSequenceClassification.from_pretrained(TL_MODEL_PATH, num_labels=6, problem_type="multi_label_classification")

        if teacher_tokenizer is None:   #If not initilized in DA phase, initialize here
            print(f"Loading teacher tokenizer...")
            tokenizer_path = BEST_TL_TEACHER_PATH / "TOKENIZER"
            if tokenizer_path.exists():
                teacher_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)   #If a saved version exists, load that one
            else:
                teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE)     #Otherwise, load the base version of the teacher model tokenizer


        ### Load evaluation dataset                
        evaluation_dataset = jsonDataset.jigsawDataset(JIGSAW_DATASET_TEST_PATH)
        label_names = evaluation_dataset.class_names

        #EVAL TEACHER MODEL
        print("\n\nEvaluating teacher model...")
        evaluation_teacher_collator = jsonDataset.SupervisedCollator(teacher_tokenizer, max_length=args.max_length_dis)
        evaluation_teacher_args = TrainingArguments(
            output_dir=BEST_TL_TEACHER_PATH / "evaluation",
            per_device_eval_batch_size=args.batch_size_tl,
            dataloader_num_workers=int(args.num_workers_tl),
            remove_unused_columns=False,
            report_to=[],
            label_names=["labels"],
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

        evaluation_path = BEST_TL_TEACHER_PATH / "evaluation"
        evaluation_path.mkdir(parents=True, exist_ok=True)
        evaluation_report = {
            "teacher_index": 1,
            "teacher_metrics": teacher_metrics,
            "teacher_benchmark": teacher_benchmark,
        }
    
        with (evaluation_path / "evaluation_results.json").open("w", encoding="utf-8") as evaluation_file:
            json.dump(evaluation_report, evaluation_file, indent=2)
        print(f"Teacher evaluation saved to {evaluation_path / 'evaluation_results.json'}")    
        teacher_model.cpu()


        #EVAL STUDENT MODELS
        if student_tokenizer is None:   #If not initilized in DA phase, initialize here
            if TOKENIZER_STUDENT_PATH.exists():
                student_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_STUDENT_PATH)   #If a saved version exists, load that one
            else:
                student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE)     #Otherwise, load the base version of the teacher model tokenizer

        student_collator = jsonDataset.SupervisedCollator(student_tokenizer)

        with open(STUDENT_NAMES_PATH, 'r') as f:    #Get student model names and paths from the JSON file for reference
            student_info = json.load(f)
            student_models_names = student_info["names"]
            student_models_paths = [Path(p) for p in student_info["paths"]]

        # Evaluate each student model
        for student_index in range(len(student_models_paths)):

            student_path = student_models_paths[student_index]                                          #Path for current student          
            if student_index < len(student_models):                                                     #If student is already loaded use them
                student_model = student_models[student_index]
            else:               
                weights_path = student_path / "model.safetensors"
                if not weights_path.exists():
                    raise FileNotFoundError(f"Student weights not found: {weights_path}")
                student_model = modelSetups.create_student_model(student_models_names[student_index],vocab_size=len(student_tokenizer))
                student_model.load_state_dict(load_file(str(weights_path), device="cpu"))
                student_model.to(device)
            
            model_name = student_models_names[student_index]
            print(f"\n\nEvaluating student model {student_index + 1}: {model_name}")

            student_evaluation_args = TrainingArguments(        
                output_dir=student_path / "evaluation",
                per_device_eval_batch_size=1,                           #Batch size is set to 1 as that's how the model will be used in production
                dataloader_num_workers=int(args.num_workers_dis),
                remove_unused_columns=False,
                report_to=[],
                label_names=["labels"],
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
                batch_size=1,                           #Batch size is set to 1 as that's how the model will be used in production,
                num_workers=int(args.num_workers_dis),
                label_names=label_names,
                model_path=student_path,
            )
            print(f"\nStudent {student_index + 1} metrics:")
            print(student_metrics)
            print(f"Student {student_index + 1} end-to-end benchmark:")
            print(student_benchmark["benchmark"])

            evaluation_path = student_path / "evaluation"
            evaluation_path.mkdir(parents=True, exist_ok=True)
            evaluation_report = {
                "student_index": student_index + 1,
                "student_metrics": student_metrics,
                "student_benchmark": student_benchmark,
            }
            
            with (evaluation_path / "evaluation_results.json").open("w", encoding="utf-8") as evaluation_file:
                json.dump(evaluation_report, evaluation_file, indent=2)
            print(f"Student evaluation saved to {evaluation_path / 'evaluation_results.json'}")
            
            student_model.cpu()


if __name__ == "__main__":
    main()



### TODO:
#
# Add better cleaning of the CPU / GPU memory after each model is trained and evaluated
# Add a phase for hyperparameter tuning of select students 
# Clean / add code that filters the jigsaw dataset to exclude "Username:" and their actual names
# Run the entire pipeline with a larger JEF1056 dataset on the ECE Cluster