#!/bin/bash
#
# These NER datasets are directly 
#reterived from BioBERT (https://github.com/dmis-lab/biobert) via this link (https://drive.google.com/file/d/1cGqvAm9IZ_86C4Mj7Zf-w9CFilYVDl8j/view) and
#BioRED dataset is publically avaliable in https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/. The datasets are used for fine-tuing from scratch

set -e

# Configure download location
DOWNLOAD_PATH="$BIOBERT_DATA"
if [ "$BIOBERT_DATA" == "" ]; then
    echo "BIOBERT_DATA not set; downloading to default path ('data')."
    DOWNLOAD_PATH="./data"
fi
DOWNLOAD_PATH_TAR="$DOWNLOAD_PATH.zip"

# Download datasets
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nHH3UYpQImQhBTei5HiTcAAFBvsfaBw0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nHH3UYpQImQhBTei5HiTcAAFBvsfaBw0" -O "$DOWNLOAD_PATH_TAR" && rm -rf /tmp/cookies.txt
unzip "$DOWNLOAD_PATH_TAR"
rm "$DOWNLOAD_PATH_TAR"
echo "Bio dataset download done!"



# BC7T2-evaluation_v3 folder is directly 
#reterived from ncbi.nlm.nih.gov via this link (https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-evaluation_v3.zip)
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-evaluation_v3.zip
unzip BC7T2-evaluation_v3.zip
rm -r BC7T2-evaluation_v3.zip

echo "BC7T2-evaluation_v3 download done!"
