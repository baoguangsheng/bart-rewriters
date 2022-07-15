# BART Rewriters with both External and Internal Sentence Extractors.

**This code is for our paper [A General Contextualized Rewriting Framework for Text Summarization](https://arxiv.org/abs/2207.05948).**

**Python Version**: Python3.7.10

**Package Requirements**: torch==1.9.0 tensorboardX numpy==1.21.6

**Framework**: Our model and experiments are built upon [fairseq v0.10.2](https://github.com/pytorch/fairseq).

Before running the scripts, please install the dependencies by:
```
    bash setup.sh
```
Please also follow the readme files under folder bart_pretrained and bert_extractors to download pretrained model and previous extractor models.
(Notes: We train our models using 2 GPUs on v100. If you want to train them on 4 GPUs, in theory you could half the number of argument --update-freq.)


In updating...

## Training

### BART-Rewriter (with external sentence extractor)
```
    CUDA_VISIBLE_DEVICES=0,1 bash exp_rewriter/run-bart-large.sh exp_test rewriter
```

### BART-JointSR (joint internal sentence selector and rewriter)
```
    CUDA_VISIBLE_DEVICES=0,1 bash exp_rewriter/run-bart-large.sh exp_test jointsr
```

## Evaluation

### BART-Rewriter (with external sentence extractor)
```
    CUDA_VISIBLE_DEVICES=0 bash exp_rewriter/test-rewriter.sh exp_test rewriter bertext
```

### BART-JointSR (joint internal sentence selector and rewriter)
```
    CUDA_VISIBLE_DEVICES=0 bash exp_rewriter/test-rewriter.sh exp_test jointsr none
```