"""
TinyML Content Moderator - code structure outline / skeleton.

    data/
        load_discord.py       # unlabeled corpus for MLM pretraining
        load_toxicity.py      # labeled dataset for classification fine-tuning
        load_twitch.py        # cross-platform eval data (scraped or existing corpus)
        preprocess.py         # shared cleaning/tokenization utilities
    models/
        teacher.py             # full BERT-based classifier
        students.py            # smaller architectures under test
    train_domain_adapt.py      # stage 1: MLM pretraining on Discord data
    train_classifier.py        # stage 2: supervised fine-tuning (teacher)
    distill.py                 # stage 3: teacher -> student distillation
    transfer_twitch.py         # stage 4: cross-platform adaptation/eval
    evaluate.py                # shared metrics: accuracy/F1, latency, size, memory
    utils.py                   # device setup, seeding, logging
"""

# ---------------------------------------------------------------------------
# 0. Shared setup (device, seeding) — same pattern as deeplearningweek2.py
# ---------------------------------------------------------------------------
import torch

def get_device():
    # Reuse exactly what you already have working in deeplearningweek2.py
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        # ... existing CUDA info printout ...
    else:
        device = torch.device('cpu')
    return device


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------
# 1a. Discord corpus (unlabeled) — for MLM domain-adaptive pretraining
#     - load raw JSON/parquet dump
#     - strip non-text content (attachments, embeds, system messages)
#     - basic cleaning (strip mentions/emoji-codes or decide to keep them —
#       document the decision either way)
#     - tokenize with the same tokenizer as your BERT checkpoint
#     - build a HuggingFace-style dataset / DataLoader for MLM objective
#       (random token masking handled by a data collator)

# 1b. Labeled toxicity dataset — for supervised classification fine-tuning
#     - load, inspect label distribution (class imbalance is likely — decide
#       on a strategy: class weights, oversampling, or accept it and report it)
#     - train/val/test split (stratified on label)
#     - tokenize, build DataLoaders (mirrors trainDataset/testDataset pattern
#       from deeplearningweek2.py, just with tokenized text instead of pixels)

# 1c. Twitch data — for cross-platform evaluation
#     - load scraped VOD chat / existing corpus
#     - if a labeled eval subset exists, keep it separate from any further
#       fine-tuning data (don't contaminate your eval set)
#     - same tokenization pipeline as above, for consistency


# ---------------------------------------------------------------------------
# 2. Model setup
# ---------------------------------------------------------------------------
# 2a. Teacher: BERT-base + classification head
#     - load pretrained checkpoint (e.g. via a transformers-style library)
#     - simple linear head on top of [CLS] / pooled output -> num_classes

# 2b. Student architectures (the actual "TinyML" part)
#     - define 2-4 candidate small architectures to compare, e.g.:
#         - reduced-layer Transformer (fewer layers/heads/hidden dim)
#         - small BiLSTM/GRU classifier (non-Transformer baseline —
#           useful contrast point, and much cheaper to train from scratch)
#     - keep a simple registry/dict so training/eval code can loop over
#       "model variants" generically rather than duplicating code per model


# ---------------------------------------------------------------------------
# 3. Stage 1 — Domain-adaptive pretraining (MLM, unsupervised, Discord data)
# ---------------------------------------------------------------------------
# - continue-pretrain teacher's base encoder on Discord corpus
# - standard MLM loss, no labels needed
# - checkpoint the adapted encoder — this becomes the starting point for stage 2


# ---------------------------------------------------------------------------
# 4. Stage 2 — Supervised fine-tuning (teacher, labeled toxicity data)
# ---------------------------------------------------------------------------
# - same train-loop shape as deeplearningweek2.py:
#     for epoch in range(nEpoch):
#         for xbatch, ybatch in trainDataLoader:
#             xbatch, ybatch = xbatch.to(device), ybatch.to(device)
#             optimizer.zero_grad()
#             y_pred = model(xbatch)
#             loss = loss_fn(y_pred, ybatch)
#             loss.backward()
#             optimizer.step()
# - track accuracy/F1 per epoch same way (accuracy[] array pattern already used)
# - this fine-tuned model is your "teacher" for distillation


# ---------------------------------------------------------------------------
# 5. Stage 3 — Distillation (teacher -> each student architecture)
# ---------------------------------------------------------------------------
# - distillation loss = weighted combination of:
#     - hard-label loss (student vs. true label, e.g. cross-entropy)
#     - soft-label loss (student vs. teacher's softened logits, e.g. KL-div
#       with a temperature parameter)
# - loop over each student architecture from the registry in step 2b
# - save each distilled student checkpoint + its size/param count


# ---------------------------------------------------------------------------
# 6. Stage 4 — Cross-platform transfer (Twitch)
# ---------------------------------------------------------------------------
# - take top 2-3 distilled students (by the Pareto results from stage 5 eval)
# - evaluate zero-shot on Twitch eval subset first (no further training)
# - optionally: light fine-tune on a small labeled Twitch subset, compare
#   zero-shot vs. fine-tuned transfer performance


# ---------------------------------------------------------------------------
# 7. Evaluation / efficiency metrics (shared across all stages)
# ---------------------------------------------------------------------------
# For every model variant, collect in one table/dict:
#   - accuracy / F1 (and per-class if imbalance is notable)
#   - parameter count
#   - model size on disk (MB)
#   - inference latency (ms/message) — measure on GPU AND at least one
#     CPU-only / low-power setting, since that's the actual point of the project
#   - peak memory during inference
#
# This table is what feeds the accuracy-vs-cost plot for the report.


# ---------------------------------------------------------------------------
# 8. Plotting (reuse matplotlib pattern from deeplearningweek2.py)
# ---------------------------------------------------------------------------
# - training curves per stage (loss/accuracy vs. epoch) — same as existing plt code
# - headline plot: accuracy/F1 (y) vs. parameter count or latency (x),
#   one point per model variant, teacher included as reference point
