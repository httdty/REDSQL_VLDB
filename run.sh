#!/bin/bash

set -e

# 检查是否提供了输入参数
if [ "$#" -le 1 ]; then
    echo "Usage: $0 model"
    exit 1
fi

model_full="$1"
model_short="${1:0:4}"
baseline="$2"
baseline_short="${baseline%%_*}"

# 执行 Python 脚本，使用提供的模型路径
python -m main.run \
  --model_name=${model_full} \
  --batch_size=2 \
  --exp_name=A_${model_short}_${baseline_short} \
  --bug_fix \
  --consistency_num=30 \
  --stage=dev \
  --preds=./datasets/preds/bird_dev/${baseline} \
  --db_content_index_path=./index/bird/db_contents_index \
  --annotation=./datasets/bird/dev_annotation_o.json \
  --output_dir=./output \
  --dev_file=./datasets/bird/dev.json \
  --table_file=./datasets/bird/dev_tables.json \
  --train_table_file= \
  --db_dir=./datasets/bird/database

output_path="./output/A_${model_short}_${baseline_short}.txt"

cp predicted_sql.txt "${output_path}"


python -m main.run \
  --model_name=${model_full} \
  --batch_size=2 \
  --exp_name=AB_${model_short}_${baseline_short} \
  --bug_fix \
  --bug_only \
  --consistency_num=30 \
  --stage=dev \
  --preds=${output_path} \
  --db_content_index_path=./index/bird/db_contents_index \
  --annotation=./datasets/bird/dev_annotation_o.json \
  --output_dir=./output \
  --dev_file=./datasets/bird/dev.json \
  --table_file=./datasets/bird/dev_tables.json \
  --train_table_file= \
  --db_dir=./datasets/bird/database

output_path="./output/AB_${model}_${baseline_short}.txt"

cp predicted_sql.txt "${output_path}"
