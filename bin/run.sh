#!/bin/bash

export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=3

python src/main.py \
    --train_json data/data/my_train.json \
    --validation_json data/data/my_validation.json \
    --test_tracks data/data/test-tracks_2.json \
    --test_queries data/data/test-queries.json \
    --output_dir outputs/duo_effcient \
    --image_model efficientnet-b0 \
    --bert_model distilroberta-base \
    --image_dir data \
    --image_size 224 \
    --max_len 64 \
    --n_hiddens 4 \
    --out_features 256 \
    --num_worker 2 \
    --lr 0.00001 \
    --batch_size 32 \
    --n_epochs 20 \
    --seed 1710 \
    --num_warmup_steps 0 \
    --patience 10 \
    --ckpt_path outputs/duo_effcient/epoch_0_mrr_0.0180.bin \
    --sample_submission data/baseline/baseline-results.json \
    --submission_path results.json \
    --do_infer