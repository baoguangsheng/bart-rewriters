import torch
from fairseq.models.bart import BARTModel
import argparse
import os.path as path
from .data_builder import match_by_rouge12L
import numpy as np
from tqdm import tqdm
from bert_extractors.src.extractors import BertSumExtractor, SentExtractor

# Extractor that uses oracle matches
class OracleExtractor:
    def extract(self, docs, refs):
        exts = []
        for doc_sents, ref_sents in zip(docs, refs):
            abs_art_scores = np.array(match_by_rouge12L(doc_sents, ref_sents))
            abs_art_idx = np.argmax(abs_art_scores, axis=1).tolist()
            exts.append(abs_art_idx)
        return exts

# Extractor that uses Lead3 sentences
class Lead3Extractor:
    def extract(self, docs, refs):
        exts = [[0, 1, 2][:len(doc)] for doc in docs]
        return exts

# Extractor that uses BERTSUMEXT model to select sentences
class BERTSUMEXTExtractor:
    def __init__(self, model_file):
        self.model = BertSumExtractor(model_file, False)

    def extract(self, docs, refs):
        batch = self.model.create_batch(docs, refs)
        exts, exts_ = self.model.extract(batch)
        # exts_str = [[doc[idx].lower() for idx in ext] for doc, ext in zip(docs, exts)]
        # assert exts_str == exts_
        return exts

# Extractor that uses BERTExt model to select sentences
class BERTExtExtractor:
    def __init__(self, model_file):
        self.model = SentExtractor(model_file)

    def extract(self, docs, refs):
        batch = self.model.create_batch(docs, refs)
        exts = self.model.extract(batch)
        return exts

# Extractor that uses BART autoregressive model to select sentences
from fairseq.data.encoders.gpt2_bpe import get_encoder
from fairseq import checkpoint_utils, options, scoring, tasks, utils

class BARTAutoregExtractor:
    def __init__(self, extractor_path):
        parser = options.get_generation_parser()
        self.args = options.parse_args_and_arch(parser, input_args=[extractor_path])
        self.args.data_format = 'jointsr'
        self.args.beam = 2
        self.args.device = 'cuda'
        self.args.source_lang = 'source'
        self.args.target_lang = 'target'
        self.args.task = 'summarization_rewriter'
        self.bpe = get_encoder(path.join(extractor_path, 'encoder.json'),
                               path.join(extractor_path, 'vocab.bpe'))
        self.task = tasks.setup_task(self.args)
        self.model, self.model_args = checkpoint_utils.load_model_ensemble(
            [path.join(extractor_path, 'checkpoint_best.pt')], task=self.task)
        self.model = self.model[0]
        self.model.make_generation_fast_(beamable_mm_beam_size=self.args.beam)
        self.model.cuda().eval()
        self.generator = self.task.build_generator([self.model], self.args)

    def encode(self, line):
        line = ' ' + line.strip()
        ids = self.bpe.encode(line)
        return list(map(str, ids))

    def encode_lines(self, doc_sents):
        slines = doc_sents
        bos = '<S%s>'
        eos = '</S>'

        # following code are copied from data_builder.py
        sids = []
        stokens = []
        if self.args.data_format == 'autoreg':  # the format required by autoregressive extractor: <S1> <S2>
            # source
            for idx, line in enumerate(slines):
                prefix = [bos % (idx + 1)]
                suffix = [eos]
                sids.extend(prefix + self.encode(line) + suffix)
                stokens.extend(prefix + line.split() + suffix)
        else:
            raise Exception()
        return sids, stokens

    def extract(self, docs, refs):
        src_dict = self.model.encoder.dictionary
        tgt_dict = self.model.decoder.dictionary
        # make a batch
        src_tokens = []
        src_lengths = []
        for doc_sents in docs:
            sids, stokens = self.encode_lines(doc_sents)
            tokens = [src_dict.bos_index] + [src_dict.index(id) for id in sids]
            tokens = tokens[:self.model_args.max_source_positions - 1] + [src_dict.eos_index]
            src_tokens.append(tokens)
            src_lengths.append(len(tokens))
        max_len = max(src_lengths)
        src_tokens = [line + [src_dict.pad_index] * (max_len - len(line)) for line in src_tokens]
        src_tokens = torch.tensor(src_tokens, device=self.args.device)
        src_lengths = torch.tensor(src_lengths, device=self.args.device)
        sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}}
        # generate
        hypos = self.generator(sample, prefix_tokens=None)
        hypos = [h[0]['tokens'].cpu().numpy().tolist() for h in hypos]
        assert len(hypos) == len(docs)
        exts = []
        for i, hypo in enumerate(hypos):
            ext = []
            for tok in hypo:
                tok = tgt_dict[tok]
                if tok in ['<s>', '</s>', '<pad>', '<unk>', '</S>']:
                    continue
                elif tok.startswith('<S') and tok.endswith('>'):
                    idx = int(tok[2:-1]) - 1
                    if idx < len(docs[i]):
                        ext.append(idx)
            exts.append(ext)
        return exts


def build_extractor(args):
    if args.extractor == 'oracle':
        return OracleExtractor()
    elif args.extractor == 'lead3':
        return Lead3Extractor()
    elif args.extractor == 'bertsumext':
        return BERTSUMEXTExtractor(path.join(args.extractor_path, 'BERTSUMEXT.pt'))
    elif args.extractor == 'bertext':
        return BERTExtExtractor(path.join(args.extractor_path, 'SentExt.model_step_25000.pt'))
    elif args.extractor in ['jointsr']:
        return BARTAutoregExtractor(args.extractor_path)
    else:
        raise Exception()

