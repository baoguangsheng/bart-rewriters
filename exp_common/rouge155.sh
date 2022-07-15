#!/usr/bin/env bash
# bash rouge155.sh reference system
ref=$1
sys=$2

echo rouge155.sh
echo `date`, Calculating ROUGE...
export CLASSPATH=exp_common/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

cat $sys | sed -e 's/ <SEP> /<q>/g' | \
java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $sys.tmp

cat $ref | sed -e 's/ <SEP> /<q>/g' | \
java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $ref.tmp

python bert_extractors/src/exp_base.py -gold $ref.tmp -candi $sys.tmp
