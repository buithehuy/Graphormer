# #!/usr/bin/env bash

# Rice Diseases Classification - Optimized for Tesla T4
export PYTHONPATH=/content/Graphormer:/content/Graphormer/examples:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 fairseq-train \
--user-dir ../../graphormer \
--num-workers 2 \
--ddp-backend=legacy_ddp \
--dataset-name rice_diseases \
--dataset-source pyg \
--task graph_prediction \
--criterion multiclass_cross_entropy \
--arch graphormer_slim \
--num-classes 4 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.1 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
--lr-scheduler polynomial_decay --power 1 \
--warmup-updates 1000 \
--total-num-update 50000 \
--lr 3e-4 --end-learning-rate 1e-9 \
--batch-size 64 \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 12 \
--encoder-embed-dim 256 \
--encoder-ffn-embed-dim 512 \
--encoder-attention-heads 8 \
--max-epoch 150 \
--save-dir ./ckpts_rice_diseases_v2 \
--patience 30