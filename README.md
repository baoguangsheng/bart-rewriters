# BART Rewriters

**This code is for our paper [A General Contextualized Rewriting Framework for Text Summarization](https://arxiv.org/abs/2207.05948).**

**Python Version**: Python3.7.10

**Package Requirements**: torch==1.9.0 tensorboardX numpy==1.21.6

**Framework**: Our model and experiments are built upon [fairseq v0.10.2](https://github.com/pytorch/fairseq).

Before running the scripts, please install the dependencies by:
```
    bash setup.sh
```

Before evaluating BART-Rewriter, please follow the readme file under folder bert_extractors to download previous BERT extractor models.

(Notes: We train our models on 2 Tesla V100.)


## Option 1: play with the trained Models

1) Download the [prepared data and trained models](https://drive.google.com/file/d/1uXDfsB3Zgio3s1Dg60h96-OPsyQrOeki/view?usp=sharing).
   Unzip the files into folder exp_test.

2) Evaluate the models:
```
    # BART-Rewriter (Rewriter with external sentence extractor)
    CUDA_VISIBLE_DEVICES=0 bash exp_rewriter/test-rewriter.sh exp_test rewriter bertext
    
    # BART-JointSR (Rewriter with joint internal sentence selector)
    CUDA_VISIBLE_DEVICES=0 bash exp_rewriter/test-rewriter.sh exp_test jointsr none
```


## Option 2: train the models from scratch 

### Prepare data:

1) Preprocess CNN/Dialy Mail.

   Follow the [instruction](https://github.com/abisee/cnn-dailymail) to convert the data into tokenized stories:
```
    cnn-dailymail/cnn_stories_tokenized/
    cnn-dailymail/dm_stories_tokenized/
```

2) Preprocess and binarize for our model:
```
    # BART-Rewriter (Rewriter with external sentence extractor)
    bash exp_rewriter/prepare-data.sh exp_test large rewriter

    # BART-JointSR (Rewriter with joint internal sentence selector)
    bash exp_rewriter/prepare-data.sh exp_test large jointsr
```

### Train the model:
```
    # BART-Rewriter (Rewriter with external sentence extractor)
    CUDA_VISIBLE_DEVICES=0,1 bash exp_rewriter/run-bart-large.sh exp_test rewriter

    # BART-JointSR (Rewriter with joint internal sentence selector)
    CUDA_VISIBLE_DEVICES=0,1 bash exp_rewriter/run-bart-large.sh exp_test jointsr
```


### Evaluate the model:
```
    # BART-Rewriter (Rewriter with external sentence extractor)
    CUDA_VISIBLE_DEVICES=0 bash exp_rewriter/test-rewriter.sh exp_test rewriter bertext
    
    # BART-JointSR (Rewriter with joint internal sentence selector)
    CUDA_VISIBLE_DEVICES=0 bash exp_rewriter/test-rewriter.sh exp_test jointsr none
```