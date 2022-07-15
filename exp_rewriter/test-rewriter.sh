#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 exp_test jointsr none"
    exit
fi

exp_path=$1
format=$2  # rewriter, jointsr
extractor=$3  # oracle, lead3, bertsumext, bertext, none
data=cnndm


echo `date`, exp_path: $exp_path, data: $data, format: $format, extractor: $extractor
tok_path=$exp_path/$data.tokenized
run_path=$exp_path/run-$format
cp_path=$run_path/$data.checkpoints
res_path=$run_path/$data.results

rm $res_path/test.${extractor}.ref -f
rm $res_path/test.${extractor}.gen -f

echo `date`, Generate summary...
mkdir -p $res_path
D=test
beam_search="--beam 2 --max-len-a 0 --max-len-b 200 --min-len 20 --lenpen 1.0 --no-repeat-ngram-size 0"

python -m exp_rewriter.rewriter $cp_path --data-format $format --source $tok_path/$D.source --target $tok_path/$D.target \
  --extractor $extractor --extractor-path 'bert_extractors/models' --rewriter-path $cp_path --outdir $res_path \
  --batch-size 8 $beam_search \
  > $run_path/test_${extractor}.$data.log 2>&1


echo `date`, Calculating ROUGE...
cp $tok_path/test.target $res_path/test.${extractor}.ref -f
bash exp_common/rouge155.sh $res_path/test.${extractor}.ref $res_path/test.${extractor}.gen >> $run_path/test_${extractor}.$data.log 2>&1
