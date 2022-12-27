#!/bin/bash
#
# These NER datasets are directly 
#reterived from BioBERT (https://github.com/dmis-lab/biobert) via this link (https://drive.google.com/file/d/1cGqvAm9IZ_86C4Mj7Zf-w9CFilYVDl8j/view) and
#BioRED dataset is publically avaliable in https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/. The datasets are used for fine-tuing from scratch

gdown https://drive.google.com/uc?id=1nHH3UYpQImQhBTei5HiTcAAFBvsfaBw0
unzip datasets.zip
rm -r datasets.zip

echo "Bio dataset download done!"


# BC7T2-evaluation_v3 folder is directly 
#reterived from ncbi.nlm.nih.gov via this link (https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-evaluation_v3.zip)
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-evaluation_v3.zip
unzip BC7T2-evaluation_v3.zip
rm -r BC7T2-evaluation_v3.zip

echo "BC7T2-evaluation_v3 download done!"
