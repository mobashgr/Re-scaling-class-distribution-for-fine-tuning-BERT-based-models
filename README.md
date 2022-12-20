# Re-scaling class distribution for fine-tuning BERT-based models
Authors: Ghadeer Mobasher*, Pedro Ruas, Francisco M. Couto, Olga Krebs, Michael Gertz and Wolfgang Müller

## Motive
Biomedical pre-trained language models (BioPLMs) have been achieving state-of-the-art results for various biomedical text mining tasks. However, prevailing fine-tuning approaches naively train BioPLMs on targeted datasets without considering the class distributions. This is problematic, especially
when dealing with imbalanced biomedical gold-standard datasets for named entity recognition (NER). Regardless of the high-performing state-of-the-art fine-tuned NER models, the training datasets include more "O" tags. Thus these models are biased towards "O" tags and misclassify biomedical entities ("B" & "I") tags. To fill the gap, we propose WELT, a cost-sensitive trainer that handles the class imbalance for the task of biomedical NER. We investigate the impact of WELT against the traditional fine-tuning approaches on mixed-domain and domain-specific BioPLMs. In addition, we examine the effect of handling the class imbalance on another downstream task which is named entity linking (NEL)

## Installation 
**Dependencies**
-	Python (>=3.6)
-	Pytorch (>=1.2.0) 
1.	Clone this GitHub repository: `git clone https://github.com/mobashgr/WELT.git`
2.	Navigate to the WELT folder and install all necessary dependencies: `python3 -m pip install -r requirements.txt` \
Note: To install appropriate torch, follow the [download instructions](https://pytorch.org/) based on your development environment.
## Data Preparation
**NER Datasets**
| Dataset 	| Source 	|
|---	|---	|
| <ul><li>NCBI-disease</li> <li>BC5CDR-disease</li>  <li>BC5CDR-chem</li></ul> 	| NER datasets are directly reterived from [BioBERT](https://github.com/dmis-lab/biobert) via this [link](https://drive.google.com/file/d/1cGqvAm9IZ_86C4Mj7Zf-w9CFilYVDl8j/view) 	|
| <ul><li>BioRED-Dis</li>  <li>BioRED-Chem</li></ul> 	| We have extended the prementioned NER datastes to include [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/). To convert from  `BioC XML / JSON` to `conll`, we used [bconv](https://github.com/lfurrer/bconv) and filtered the chemical and disease entities. 	|

**Data Download** \
To directly download NER datasets, use `download.sh` or manually download them via this [link](https://drive.google.com/file/d/1nHH3UYpQImQhBTei5HiTcAAFBvsfaBw0/view) in `WELT` directory, `unzip datasets.zip` and `rm -r datasets.zip`

**Data Pre-processing** \
We adapted the `preprocessing.sh` from [BioBERT](https://github.com/dmis-lab/biobert) to include [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/)

## Fine-tuning with handling the class imbalance
We have conducted expirements on two different BERT models using WELT weighting scheme. We have compared WELT againest the corresponding traditional fine-tuning approaches(i.e normal BioBERT fine-tuning)

**Fine-tuning BERT Models** \
Our expiremental work focused on BioBERT(mixed/continual pre-trained language model) & PubMedBERT(domain-specific/trained from scratch pre-trained language model), however WELT can be adapted to other transformers like ELECTRA.
| Model 	| Used version in HF :hugs: |
|---	|---	|
|BioBERT| [model_name_or_path](https://huggingface.co/dmis-lab/biobert-v1.1)|
|PubMedBERT| [model_name_or_path](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)|


**WELT equations** \
These equations are applied to "O" (major class), "B" & "I" (minor classes) as weighting scheme to handle class imbalance.
  $$CW_c= \textstyle 1- \dfrac{ClassDistibution_c}{TotalOfClassesDistributions_t}$$ \
  $$\text where \ c= |O| or |B| or|I| \text{and} \ t=|O|+|B|+|I|$$ 
  $$WV= \sum_{c=1}^{t} CW_c$$\
  $$\sigma \vec{(WV)} i=\dfrac {e^{WV_i}}{\sum\limits_{c=1}^{t} e^{WV_c}}$$\
 $$loss(x,class)=\textstyle \sigma \vec{(WV)} i [class] \Theta$$ \
 $$where,\Theta= -x[class]+\log{\sum_j exp(x[j])}$$  

**Cost-Sensitive Fine-Tuning**

We have adapted [BioBERT-run_ner.py](https://github.com/dmis-lab/biobert-pytorch/blob/master/named-entity-recognition/run_ner.py) to develop in [run_weight_scheme.py](https://github.com/mobashgr/Re-scaling-class-distribution-for-fine-tuning-BERT-based-models/blob/main/named-entity-recognition/run_weight_scheme.py#L94-103) that extends `Trainer` class to `WeightedLossTrainer` and override `compute_loss` function to include [BioBERT-run_ner.py](https://github.com/dmis-lab/biobert-pytorch/blob/master/named-entity-recognition/run_ner.py) 
[WELT](https://github.com/mobashgr/Re-scaling-class-distribution-for-fine-tuning-BERT-based-models/blob/main/named-entity-recognition/run_weight_scheme.py#L129-142) in [weighted Cross-Entropy loss function] (https://github.com/mobashgr/Re-scaling-class-distribution-for-fine-tuning-BERT-based-models/blob/main/named-entity-recognition/run_weight_scheme.py#L101). 

**Evaluation** \
For fair comparison we have used the same NER evaluation approach of [BioBERT](https://github.com/dmis-lab/biobert)

**Usage Example** \
This is an example for fine-tuning `BioRED-Chem` over `SciBERT` using `ENS` weight scheme with $\beta$ of 0.3
```bash
cd named-entity-recognition
./preprocess.sh

export SAVE_DIR=./output
export DATA_DIR=../datasets/NER

export MAX_LENGTH=384
export BATCH_SIZE=5
export NUM_EPOCHS=20
export SAVE_STEPS=1000
export ENTITY=BioRED-Chem
export SEED=1

python run_weight_scheme.py \
    --data_dir ${DATA_DIR}/${ENTITY}/ \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path allenai/scibert_scivocab_uncased \
   --output_dir ${ENTITY}-WLT-${MAX_LENGTH}-SciBERT-ENS-4 \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --weight_scheme ENS \
    --beta_factor 0.3  \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir
  ```
## Fine-tuned models avaliable on HF :hugs:
 ## Citation
  (TBD)
## Acknowledgment
Ghadeer Mobasher is part of the PoLiMeR-ITN (http://polimer-itn.eu/) and is supported by European Union’s Horizon 2020 research and innovation program under the Marie Skłodowska-Curie grant agreement PoLiMeR, No 81261
