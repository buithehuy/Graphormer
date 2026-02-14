# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#!/usr/bin/env bash

# Rice Diseases Classification Training Script
# Dataset: 4 classes (BrownSpot, Healthy, Hispa, LeafBlast)
# Preprocessed graphs stored as individual .pt files

# Add paths for imports
# export PYTHONPATH=/content/Graphormer:/content/Graphormer/examples:$PYTHONPATH



CUDA_VISIBLE_DEVICES=0 fairseq-train \
--user-dir ../../graphormer \
--num-workers 2 \
--ddp-backend=legacy_ddp \
--dataset-name rice_diseases \
--dataset-source pyg \
--task graph_prediction \
--criterion multiclass_cross_entropy \
--arch graphormer_base \
--num-classes 4 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.1 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
--lr-scheduler polynomial_decay --power 1 \
--warmup-updates 5000 \
--total-num-update 50000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size 32 \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 12 \
--encoder-embed-dim 128 \
--encoder-ffn-embed-dim 512 \
--encoder-attention-heads 8 \
--max-epoch 100 \
--save-dir ./ckpts_rice_diseases \
--patience 20
