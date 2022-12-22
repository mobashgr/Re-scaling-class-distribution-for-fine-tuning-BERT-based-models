## Hyperparameters
The table shows the used hyper-parameters for fine-tuning BERT models.

| Dataset | Epochs | Batch Size | Maximum Length |
| :------------------ | :----- | :----- | :----- |
| NCBI-Disease | 10| 8 | 320 |
| BC5CDR-Disease |10 | 8 |320|
| BioRED-Disease | 10 | 8 | 384 |
| BC5CDR-Chemical | 10 | 16 | 256 |
| BioRED-Chem | 10 | 32| 128 |

## Fine-tuned HF 🤗
| Dataset | Orignal| WELT 
| :------------------ | :----- | :----- 
| NCBI-Disease | [model_name_or_path](https://huggingface.co/ghadeermobasher/Original-BioBERT-NCBI)| [model_name_or_path](https://huggingface.co/ghadeermobasher/WELT-BioBERT-NCBI)|
| BC5CDR-Disease |[model_name_or_path](https://huggingface.co/ghadeermobasher/Original-BioBERT-BC5CDRDisease) | [model_name_or_path](https://huggingface.co/ghadeermobasher/WELT-BioBERT-BC5CDRDisease) |
| BioRED-Disease |[model_name_or_path](https://huggingface.co/ghadeermobasher/Original-PubMedBERT-BioRedDis) | [model_name_or_path](https://huggingface.co/ghadeermobasher/WELT-PubMedBERT-BioRedDis) |
| BC5CDR-Chemical |[model_name_or_path](https://huggingface.co/ghadeermobasher/Original-PubMedBERT-BC5CDRChemical) | [model_name_or_path](https://huggingface.co/ghadeermobasher/WELT-PubMedBERT-BC5CDRChemical) |
| BioRED-Chem |[model_name_or_path](https://huggingface.co/ghadeermobasher/Original-PubMedBERT-BioRedChemical) | [model_name_or_path](https://huggingface.co/ghadeermobasher/WELT-PubMedBERT-BioRedChemical) |

## Usage example for WELT finetuning
This is an example of fine-tuning `NCBI-Disease` over `BioBERT` using an `WELT` weight scheme
```bash

cd named-entity-recognition
./preprocess.sh

export SAVE_DIR=./output
export DATA_DIR=../datasets/NER

export MAX_LENGTH=320
export BATCH_SIZE=8
export NUM_EPOCHS=10
export SAVE_STEPS=1000
export ENTITY=NCBI-disease
export SEED=1

python run_weight_scheme.py \
    --data_dir ${DATA_DIR}/${ENTITY}/ \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path dmis-lab/biobert-v1.1 \
   --output_dir ${ENTITY}-WELT-${MAX_LENGTH}-BioBERT \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --weight_scheme WELT \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir
  ```
## Usage example for predicting disease entities using WELT
This is an example of fine-tuning `NCBI-Disease` over `BioBERT` using an `WELT` fine-tuned model on HF
```bash
cd named-entity-recognition

python3 ner.py \
--xmlfilepath Re-scaling-class-distribution-for-fine-tuning-BERT-based-models/unannotatedxmls/NCBItestset_corpus_noannotations.xml \
--model_name_or_path ghadeermobasher/WELT-BioBERT-NCBI \
--NERType Disease \
--outputfilepath Re-scaling-class-distribution-for-fine-tuning-BERT-based-models/predictedpath/NCBI-WELT-BioBERT-example.xml
  ```
## Usage example for strict evaluation of `NCBI-Disease` predicted file using WELT

Assuming that [`download.sh`](https://github.com/mobashgr/Re-scaling-class-distribution-for-fine-tuning-BERT-based-models/blob/main/download.sh)has been already executed.

```bash
cd BC7T2-evaluation_v3

python3 evaluate.py \
--reference_path Re-scaling-class-distribution-for-fine-tuning-BERT-based-models/referencepath/NCBItestset_corpus.xml \
 --prediction_path Re-scaling-class-distribution-for-fine-tuning-BERT-based-models/predictedpath/NCBI-WELT-BioBERT.xml \
 --evaluation_type span \
 --evaluation_method strict \
 --annotation_type Disease
 ```

