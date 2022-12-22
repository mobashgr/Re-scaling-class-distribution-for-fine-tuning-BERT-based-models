## Hyperparameters
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
This is an example of fine-tuning `NCBI-Disease` over `BioBERT` using an `WELT` weight scheme
```bash
cd named-entity-recognition

python3 ner.py --xmlfilepath Re-scaling-class-distribution-for-fine-tuning-BERT-based-models/unannotatedxmls/CDR_TestSet.BioC_Disease_noannotations.xml 
--model_name_or_path ghadeermobasher/Original-BioBERT-BC5CDRDisease   --NERType Disease --outputfilepath /hits/basement/sdbv/mobashgr/testfiles/BC7T2-evaluation_v3/drive-download-20220131T174025Z-001/BC7T2-NLMChem-corpus_v2.BioC.xml/PredictedPath/job_ghadeermobasher/Original-BioBERT-BC5CDRDisease.xml
  ```
