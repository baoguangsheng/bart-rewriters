#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 exp_test large jointsr"
    exit
fi

exp_path=$1
model_size=$2  # base, large
format=$3  # rewriter, jointsr
data=cnndm
bart_path=bart.$model_size

echo `date`, exp_path: $exp_path, data: $data
tok_path=$exp_path/$data.tokenized
seg_path=$exp_path/$data.segmented_$format
enc_path=$exp_path/$data.encoded_$format
bin_path=$exp_path/$data.binarized_$format

echo `date`, Prepraring tokenized data...
python exp_common/make_datafiles.py --dataset $data --sep ' <SEP> ' --res $tok_path

echo `date`, Prepraring segmented data...
for D in valid train; do
  python exp_rewriter/data_builder.py --workers 12 \
         --encoder-json $bart_path/encoder.json --vocab-bpe $bart_path/vocab.bpe \
         --data-format $format --source $tok_path/$D.source --target $tok_path/$D.target \
         --outdir-tok $seg_path --outdir-id $enc_path
done

echo `date`, Prepraring binarized data...
mkdir -p $bin_path
cp $bart_path/dict.txt $bin_path/dict.txt -f
cat exp_rewriter/dict_tags.txt >> $bin_path/dict.txt
python -m fairseq_cli.preprocess --source-lang source --target-lang target --workers 12 \
       --trainpref $enc_path/train --validpref $enc_path/valid  \
       --destdir $bin_path --srcdict $bin_path/dict.txt --tgtdict $bin_path/dict.txt
