#!/bin/bash
# Run all 8 experiments sequentially with DeepSeek-V3 via chatanywhere
export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
export CUDA_VISIBLE_DEVICES=4

MODEL="deepseek-v3"

cd ~/RAG-Proj

echo "=== [1/8] FIRE FCB ===" && date
/home1/pzm/miniconda3/bin/python fire/run_fire_ddg.py --dataset factcheckbench --model openai:$MODEL
echo "=== [2/8] FIRE FacTool ===" && date
/home1/pzm/miniconda3/bin/python fire/run_fire_ddg.py --dataset factool_qa --model openai:$MODEL
echo "=== [3/8] FIRE FELM ===" && date
/home1/pzm/miniconda3/bin/python fire/run_fire_ddg.py --dataset felm_wk --model openai:$MODEL
echo "=== [4/8] FIRE BingCheck ===" && date
/home1/pzm/miniconda3/bin/python fire/run_fire_ddg.py --dataset bingcheck --model openai:$MODEL

cd ~/RAG-Proj/SWIFT

echo "=== [5/8] SWIFT FCB ===" && date
/home1/pzm/miniconda3/bin/python inference/inference.py \
    --experiment_name swift_v4_dsv3_t07_fcb_ddg \
    --model $MODEL \
    --input_path data/factcheckbench/test.csv \
    --critic_path checkpoints/swift_v4 --threshold 0.7 --search_engine ddg
echo "=== [6/8] SWIFT FacTool ===" && date
/home1/pzm/miniconda3/bin/python inference/inference.py \
    --experiment_name swift_v4_dsv3_t07_factool_ddg \
    --model $MODEL \
    --input_path data/factool_qa/raw.csv \
    --critic_path checkpoints/swift_v4 --threshold 0.7 --search_engine ddg
echo "=== [7/8] SWIFT FELM ===" && date
/home1/pzm/miniconda3/bin/python inference/inference.py \
    --experiment_name swift_v4_dsv3_t07_felm_ddg \
    --model $MODEL \
    --input_path data/felm_wk/raw.csv \
    --critic_path checkpoints/swift_v4 --threshold 0.7 --search_engine ddg
echo "=== [8/8] SWIFT BingCheck ===" && date
/home1/pzm/miniconda3/bin/python inference/inference.py \
    --experiment_name swift_v4_dsv3_t07_bingcheck_ddg \
    --model $MODEL \
    --input_path data/bingcheck/raw.csv \
    --critic_path checkpoints/swift_v4 --threshold 0.7 --search_engine ddg

echo "=== ALL DONE ===" && date
