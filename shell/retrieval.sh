#!usr/env/bin bash

OCR_TYPE=$1
RET_TYPE=$2

python quick_start.py \
  --model_name 'mock' \
  --retriever ${RET_TYPE}  \
  --retrieve_top_k 2 \
  --data_path data/qas_v2.json \
  --docs_path data/retrieval_base/${OCR_TYPE} \
  --ocr_type ${OCR_TYPE} \
  --task 'Retrieval' \
  --evaluation_stage 'retrieval' \
  --num_threads 8 \
  --show_progress_bar True
  