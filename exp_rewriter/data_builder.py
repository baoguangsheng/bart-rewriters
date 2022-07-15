#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
import os

from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe import get_encoder

from rouge import Rouge
import numpy as np
import os.path as path
from tqdm import tqdm

''' Algorithm for matching closest sentence in article for each summary sentence
'''
def match_by_rouge12(article, abstract):
    rouge = Rouge(metrics=["rouge-1", "rouge-2"])
    res = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        scores = rouge.get_scores(hyps, refs)
        recalls = [(score["rouge-1"]["r"] + score["rouge-2"]["r"]) / 2 for score in scores]
        res.append(recalls)
    return res

def match_by_rougeL(article, abstract):
    rouge = Rouge(metrics=["rouge-l"])
    res = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        scores = rouge.get_scores(hyps, refs)
        recalls = [score["rouge-l"]["r"] for score in scores]
        res.append(recalls)
    return res

def match_by_rouge12L(article, abstract):
    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    res = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        try:
            scores = rouge.get_scores(hyps, refs)
            recalls = [(score["rouge-1"]["r"] + score["rouge-2"]["r"] + score["rouge-l"]["r"]) / 3 for score in scores]
        except Exception as ex:
            print(ex)
            print('hyps:', hyps)
            print('refs:', refs)
            recalls = [0 for _ in range(len(refs))]
        res.append(recalls)
    return res


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        line = ' ' + line.strip()
        ids = bpe.encode(line)
        return list(map(str, ids))

    def encode_lines(self, lines):
        assert len(lines) == 2, 'Lines: %s' % lines
        source, target = [line.strip() for line in lines]
        if len(source) == 0 or len(target) == 0:
            return ["EMPTY", None]

        sep = ' <SEP> '
        bos = '<S%s>'
        eos = '</S>'

        # match summary to source
        slines = source.split(sep)
        tlines = target.split(sep)
        abs_art_scores = np.array(match_by_rouge12L(slines, tlines))
        abs_art_idx = np.argmax(abs_art_scores, axis=1).tolist()

        # encode source and target
        sids = []
        stokens = []
        tids = []
        ttokens = []

        if self.args.data_format == 'rewriter':  # the format required by previous BERT ContextRewriter
            # source
            for idx, line in enumerate(slines):
                prefix = [bos % (tidx + 1) for tidx, sidx in enumerate(abs_art_idx) if sidx == idx]
                suffix = [eos] * len(prefix)
                sids.extend(prefix + self.encode(line) + suffix)
                stokens.extend(prefix + line.split() + suffix)
            # target
            for idx, line in enumerate(tlines):
                prefix = [bos % (idx + 1)]
                tids.extend(prefix + self.encode(line))
                ttokens.extend(prefix + line.split())
        elif self.args.data_format == 'jointsr':  # the format required by joint rewriter: <S1> ... <S2> ...
            # source
            for idx, line in enumerate(slines,):
                prefix = [bos % (idx + 1)]
                suffix = [eos]
                sids.extend(prefix + self.encode(line) + suffix)
                stokens.extend(prefix + line.split() + suffix)
            # target
            for idx, line in enumerate(tlines):
                sidx = abs_art_idx[idx]
                prefix = [bos % (sidx + 1)]
                tids.extend(prefix + self.encode(line))
                ttokens.extend(prefix + line.split())
        else:
            raise Exception()

        # output
        enc_lines = [sids, tids, stokens, ttokens]
        enc_lines = [' '.join(line) for line in enc_lines]
        return ["PASS", enc_lines]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        default='../shared/bart.base/encoder.json',
        help='path to encoder.json',
    )
    parser.add_argument(
        "--vocab-bpe",
        default='../shared/bart.base/vocab.bpe',
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--data-format",
        default='rewriter',
        choices=['rewriter', 'jointsr'],
        help="Data format for different experiments",
    )
    parser.add_argument(
        "--source",
        default='exp_test/cnndm.tokenized/valid.source',
        help="source file to filter/encode",
    )
    parser.add_argument(
        "--target",
        default='exp_test/cnndm.tokenized/valid.target',
        help="target file to filter/encode",
    )
    parser.add_argument(
        "--outdir-tok",
        default='exp_test/cnndm.segmented',
        help="output directory to save the encoded source/target files",
    )
    parser.add_argument(
        "--outdir-id",
        default='exp_test/cnndm.encoded',
        help="output directory to save the encoded source/target files",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if not path.exists(args.outdir_tok):
        os.mkdir(args.outdir_tok)

    if not path.exists(args.outdir_id):
        os.mkdir(args.outdir_id)

    with contextlib.ExitStack() as stack:
        inputs = [stack.enter_context(open(args.source, "r", encoding="utf-8")),
                  stack.enter_context(open(args.target, "r", encoding="utf-8"))]
        outputs = [stack.enter_context(open(path.join(args.outdir_id, path.basename(args.source)), "w", encoding="utf-8")),
                   stack.enter_context(open(path.join(args.outdir_id, path.basename(args.target)), "w", encoding="utf-8")),
                   stack.enter_context(open(path.join(args.outdir_tok, path.basename(args.source)), "w", encoding="utf-8")),
                   stack.enter_context(open(path.join(args.outdir_tok, path.basename(args.target)), "w", encoding="utf-8"))
                   ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(tqdm(encoded_lines), start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)

if __name__ == "__main__":
    main()
