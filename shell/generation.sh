#!usr/env/bin bash

OCR_TYPE=$1
LLM_TYPE=$2

python quick_start.py \
  --model_name ${LLM_TYPE} \
  --temperature 0 \
  --max_new_tokens 1280 \
  --retriever 'page' \
  --data_path data/qas_v2.json \
  --docs_path data/retrieval_base/${OCR_TYPE} \
  --ocr_type ${OCR_TYPE} \
  --task 'QA' \
  --evaluation_stage 'generation' \
  --num_threads 8 \
  --show_progress_bar True 
