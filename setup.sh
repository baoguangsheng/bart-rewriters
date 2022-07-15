#!/usr/bin/env bash

pip install --editable .

# for running of ROUGE-1.5.5
apt-get update
apt-get install -y default-jre
apt-get install -y openjdk-11-jre-headless
apt-get install -y openjdk-8-jre-headless

# for running of BERTSUMEXT and BERTEXT
pip install pytorch-transformers==1.2.0
pip install transformers==4.8.2
