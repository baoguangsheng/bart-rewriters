#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 exp_test jointsr"
    exit
fi

exp_path=$1
format=$2  # rewriter, jointsr
data=cnndm
model_size=large  # base, large
bart_path=bart_pretrained/bart.$model_size

echo `date`, data: $data, exp_path: $exp_path
tok_path=$exp_path/$data.tokenized
bin_path=$exp_path/$data.binarized_$format

run_path=$exp_path/run-$format
cp_path=$run_path/$data.checkpoints

echo `date`, run path: $run_path
mkdir -p $run_path

echo `date`, Training model...
update_freq=8

# running on 2 GPUs
python train.py  $bin_path --save-dir $cp_path --tensorboard-logdir $cp_path --seed 666 --num-workers 8 --fp16 \
       --task summarization_rewriter --arch transformer_rewriter_${model_size} --source-lang source --target-lang target --truncate-source \
       --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --skip-invalid-size-inputs-valid-test \
       --criterion label_smoothed_cross_entropy_rewriter --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.00 --clip-norm 0.1  \
       --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 500 --no-epoch-checkpoints --no-last-checkpoints \
       --max-tokens 4096 --update-freq ${update_freq} --validate-interval 1 --patience 2 --find-unused-parameters \
       --restore-file $bart_path/model.pt --reset-optimizer --reset-dataloader --reset-meters \
       --data-format $format --required-batch-size-multiple 1 \
       > $run_path/train.$data.log 2>&1

# copy encoder
cp $bart_path/encoder.json $cp_path/. -f
cp $bart_path/vocab.bpe $cp_path/. -f
cp $bin_path/dict.* $cp_path/. -f
