# BART Rewriters with both External and Internal Sentence Extractors.
Code base for the paper "[A General Contextualized Rewriting Framework for Text Summarization](https://arxiv.org/abs/2207.05948)"

Updating...

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