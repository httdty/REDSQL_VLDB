# REDSQL_SIGMOD

This README provides instructions on how to set up the environment, download the dataset, and run the REDSQL.

## Dataset Download

The dataset is organized into several folders as shown below:

```
.
├── bird
│   ├── database
│   ├── dev_annotation.json
│   ├── dev.json
│   ├── dev.sql
│   └── dev_tables.json
├── preds
│   └── Predicted_SQLs
├── spider
│   ├── database
│   ├── dev_annotation.json
│   ├── dev_gold.sql
│   ├── dev.json
│   └── dev_tables.json
├── spider_dk
├── spider_realistic
└── spider_syn
```

- bird: Contains the database, dev_annotation.json, dev.json, dev.sql, and dev_tables.json files for the bird dataset.
- preds: Contains the Predicted_SQLs folder.
- spider: Contains the database, dev_annotation.json, dev_gold.sql, dev.json, and dev_tables.json files for the spider dataset.
- spider_dk, spider_realistic, spider_syn: Additional datasets.


## Environment Build

Follow these steps to set up the environment:

1. Update the system:

   ```bash
   sudo apt-get update
   sudo apt-get install -y openjdk-11-jdk
   ```

2. Create and activate a Conda environment:

   ```bash
   conda create -n red python=3.9
   source activate red
   ```

3. Install PyTorch and other dependencies:

   ```bash
   conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
   pip install -r requirements.txt
   ```

## Command


Run the REDSQL using the following command:

```bash
python -m main.run \
  --model_name=model_name \
  --batch_size=2 \
  --exp_name=exp_name \
  --bug_fix \
  --consistency_num=30 \
  --stage=dev \
  --preds=/path/to/predicted/sql.txt \
  --db_content_index_path=/path/to/db/content/index \
  --annotation=/path/to/dev_annotation.json \
  --output_dir=./output \
  --dev_file=/path/to/dev.json \
  --table_file=/path/to/dev_tables.json \
  --db_dir=/path/to/database
```


- `--model_name`: Specify the name of the LLM.
- `--batch_size`: Set the batch size (e.g., 2).
- `--exp_name`: Name of the experiment.
- `--bug_fix`: Include this flag to apply bug fixes.
- `--consistency_num`: Set the number of consistency checks (e.g., 30).
- `--stage`: Set the stage (e.g., `dev`).
- `--preds`: Path to the file containing the predicted SQL statements (e.g., `/path/to/predicted/sql.txt`).
- `--db_content_index_path`: Path to the database content index (e.g., `/path/to/db/content/index`).
- `--annotation`: Path to the annotation file (e.g., `/path/to/dev_annotation.json`).
- `--output_dir`: Directory to save the output (e.g., `./output`).
- `--dev_file`: Path to the development file (e.g., `/path/to/dev.json`).
- `--table_file`: Path to the table file (e.g., `/path/to/dev_tables.json`).
- `--db_dir`: Directory containing the database (e.g., `/path/to/database`).

Ensure all required files are correctly placed and paths are correctly set before running the command.

