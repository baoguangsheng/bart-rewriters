#!/usr/bin/env bash

# download BART large
if [ -d "bart.large" ]; then
  echo "bart.large already exists, skipping download"
else
  echo "Downloading bart.large ..."
  wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
  tar vxf bart.large.tar.gz
fi

# download CNN/Daily Mail preprocessing code
if [ -d "cnn-dailymail" ]; then
  echo "cnn-dailymail already exists, skipping download"
else
  echo "Downloading cnn-dailymail ..."
  git clone https://github.com/abisee/cnn-dailymail.git
fi

# for running of ROUGE-1.5.5
echo "Installing JDK for running of ROUGE-1.5.5 ..."
apt-get update
apt-get install -y default-jre
apt-get install -y openjdk-11-jre-headless
apt-get install -y openjdk-8-jre-headless

# for running of BERTSUMEXT and BERTEXT
echo "Installing transformers for running BERT extractors ..."
pip install pytorch-transformers==1.2.0
pip install transformers==4.8.2

# setup fairseq
echo "Installing fairseq ..."
pip install --editable .