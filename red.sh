mkdir output
mkdir logs

python -m pre_processing.build_contents_index --output_dir=./index/bird/db_contents_index/ --db_dir=./datasets/bird/dev_database/

python -m pre_processing.doc --model_name=gpt-4-32k --output_file=./annotation.json --table_file=./datasets/bird/dev_tables.json --db_dir=./datasets/bird/dev_database/

python -m main.run --model_name=gpt-4-32k --batch_size=2 --exp_name=ALL --bug_fix --consistency_num=30 --stage=dev --preds=XXXXXXXXXXXX --db_content_index_path=./index/bird/db_contents_index --annotation=./annotation.json --output_dir=./output --dev_file=./datasets/bird/dev.json --table_file=./datasets/bird/dev_tables.json --train_table_file= --db_dir=./datasets/bird/dev_database

cp predicted_sql.txt ALL.txt

python -m main.run --model_name=gpt-4-32k --batch_size=2 --exp_name=ALL_SYN --bug_fix --bug_only --consistency_num=30 --stage=dev --preds=ALL.txt --db_content_index_path=./index/bird/db_contents_index --annotation=./annotation.json --output_dir=./output --dev_file=./datasets/bird/dev.json --table_file=./datasets/bird/dev_tables.json --train_table_file= --db_dir=./datasets/bird/dev_database