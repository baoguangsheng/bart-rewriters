import bisect
import os
import torch
import os.path as path
import logging
import argparse

import sys
src_path = path.dirname(__file__)
sys.path.append(src_path)

from others.tokenization import BertTokenizer
from exp_base import ExtDecider

logger = logging.getLogger(__name__)

''' Take document and extractive summary as input, generate abstractive summary.
'''
class Batch(object):
    def __init__(self, srcs, segs, clss, src_str, tgt_str, src_idxs, device=None):
        """Create a Batch from a list of examples."""
        self.batch_size = len(srcs)
        max_len = max([len(src) for src in srcs])
        srcs = [src + [0] * (max_len - len(src)) for src in srcs]
        segs = [seg + [0] * (max_len - len(seg)) for seg in segs]
        max_len = max([len(cls) for cls in clss])
        clss = [cls + [-1] * (max_len - len(cls)) for cls in clss]
        srcs = torch.tensor(srcs)
        segs = torch.tensor(segs)
        mask_src = 1 - (srcs == 0).int()
        clss = torch.tensor(clss)
        mask_cls = 1 - (clss == -1).int()
        clss[clss == -1] = 0

        setattr(self, 'src', srcs.to(device))
        setattr(self, 'segs', segs.to(device))
        setattr(self, 'mask_src', mask_src.to(device))
        setattr(self, 'clss', clss.to(device))
        setattr(self, 'mask_cls', mask_cls.to(device))
        setattr(self, 'src_str', src_str)
        setattr(self, 'tgt_str', tgt_str)
        setattr(self, 'src_idxs', src_idxs)

    def __len__(self):
        return self.batch_size

class Extractor:
    def __init__(self, args):
        self.args = args
        # prepare tokenizer and predictor
        self.tokenizer = BertTokenizer.from_pretrained(path.join(self.args.bert_model_path, self.model_ext.bert.model_name), do_lower_case=True)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def _preprocess(self, doc_lines):
        # Same logic as BertData to process raw input
        src = [line.lower().split() for line in doc_lines]
        src_idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in src_idxs]
        src = src[:self.args.max_src_nsents]
        src_txt = [' '.join(sent) for sent in src]

        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            segments_ids += s * [i + 1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        max_sent_id = bisect.bisect_left(cls_ids, self.args.max_pos)
        cls_ids = cls_ids[:max_sent_id]
        # Generate batch
        src_subtoken_idxs = src_subtoken_idxs[:-1][:self.args.max_pos - 1] + src_subtoken_idxs[-1:]
        segments_ids = segments_ids[:self.args.max_pos]
        return src_subtoken_idxs, segments_ids, cls_ids, src_txt, src_idxs

    def create_batch(self, docs, refs):
        srcs = []
        segs = []
        clss = []
        src_str = []
        tgt_str = []
        src_idxs = []
        for doc_lines, sum_lines in zip(docs, refs):
            src, seg, cls, src_txt, src_idx = self._preprocess(doc_lines)
            srcs.append(src)
            segs.append(seg)
            clss.append(cls)
            src_str.append(src_txt)
            tgt_str.append(' '.join(sum_lines))
            src_idxs.append(src_idx)
        return Batch(srcs, segs, clss, src_str, tgt_str, src_idxs, device=self.args.device)

class BertSumExtractor(Extractor):
    def __init__(self, ext_model_file, block_trigram=True):
        from presumm import model_builder as presumm_model
        from presumm import trainer_ext as presumm_trainer_ext
        args = self._build_ext_args()
        args.block_trigram = block_trigram
        checkpoint = torch.load(ext_model_file, map_location=lambda storage, loc: storage)
        self.name = 'BERTSUMEXT_blocktrigram' if block_trigram else 'BERTSUMEXT_noblocktrigram'
        self.model_file = ext_model_file
        self.model_ext = presumm_model.ExtSummarizer(args, args.device, checkpoint)
        self.model_ext.eval()
        self.trainer = presumm_trainer_ext.build_trainer(args, args.device_id, self.model_ext, None)
        super().__init__(args)

    def _build_ext_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-task", default='ext')
        parser.add_argument("-encoder", default='bert')
        parser.add_argument("-mode", default='train')
        parser.add_argument("-bert_model_path", default=src_path + '/../bert_pretrained/')
        parser.add_argument("-bert_data_path", default='./bert_data/cnndm')
        parser.add_argument("-model_path", default='./models/')
        parser.add_argument("-result_path", default='./results/cnndm')
        parser.add_argument('-log_file', default='./logs/cnndm.log')
        parser.add_argument("-temp_dir", default='./temp')
        parser.add_argument("-train_from", default='')

        parser.add_argument('-min_src_nsents', default=3, type=int)
        parser.add_argument('-max_src_nsents', default=100, type=int)
        parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
        parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)

        parser.add_argument("-max_pos", default=512, type=int)
        parser.add_argument("-max_tgt_len", default=140, type=int)
        parser.add_argument("-max_n_tags", default=6, type=int)
        parser.add_argument("-use_interval", default=True, type=bool)
        parser.add_argument("-large", default=False, type=bool)

        parser.add_argument("-ext_dropout", default=0.2, type=float)
        parser.add_argument("-ext_layers", default=2, type=int)
        parser.add_argument("-ext_hidden_size", default=768, type=int)
        parser.add_argument("-ext_heads", default=8, type=int)
        parser.add_argument("-ext_ff_size", default=2048, type=int)

        parser.add_argument("-param_init", default=0, type=float)
        parser.add_argument("-param_init_glorot", default=True, type=bool)
        parser.add_argument("-optim", default='adam')
        parser.add_argument("-lr", default=2e-3, type=float)
        parser.add_argument("-beta1", default=0.9, type=float)
        parser.add_argument("-beta2", default=0.999, type=float)
        parser.add_argument("-max_grad_norm", default=0, type=float)

        parser.add_argument("-train_steps", default=40000, type=int)
        parser.add_argument("-warmup_steps", default=10000, type=int)
        parser.add_argument("-report_every", default=50, type=int)
        parser.add_argument("-test_start_from", default=10000, type=int)
        parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
        parser.add_argument("-batch_size", default=8*512, type=int)
        parser.add_argument("-accum_count", default=2, type=int)
        parser.add_argument('-visible_gpus', default='0,1', type=str)
        parser.add_argument("-test_batch_size", default=8*512, type=int)

        parser.add_argument("-finetune_bert", default=True, type=bool)
        parser.add_argument('-gpu_ranks', default='0', type=str)
        parser.add_argument('-seed', default=666, type=int)

        parser.add_argument("-test_all", default=True, type=bool)
        parser.add_argument("-test_from", default='')
        parser.add_argument("-recall_eval", default=False, type=bool)
        parser.add_argument("-report_rouge", default=True, type=bool)
        parser.add_argument("-block_trigram", default=True, type=bool)

        args = parser.parse_args('')
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
        args.world_size = len(args.gpu_ranks)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
        args.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        args.device_id = 0 if args.device == "cuda" else -1
        return args

    def extract(self, batch):
        exts, srctag = self.trainer.predict(batch)
        exts = [ext[0].split('<q>') for ext in exts]
        doc_exts = []
        for src_sents, src_idx, ext_sents in zip(batch.src_str, batch.src_idxs, exts):
            doc_exts.append([src_idx[src_sents.index(sent)] for sent in ext_sents])
        # doc_exts = [list(sorted(ext)) for ext in doc_exts]
        return doc_exts, exts

class SentExtractor(Extractor):
    def __init__(self, ext_model_file):
        import models.model_builder as model
        import models.trainer_ext as trainer_ext
        args = self._build_ext_args()
        checkpoint = torch.load(ext_model_file, map_location=lambda storage, loc: storage)
        self.name = 'BERT-Ext'
        self.model_file = ext_model_file
        self.model_ext = model.ExtSummarizer(args, args.device, checkpoint)
        self.model_ext.eval()
        self.decider = ExtDecider(logger)
        self.decider.load(ext_model_file + '.config')
        self.trainer = trainer_ext.build_trainer(args, args.device_id, self.model_ext, None)
        super().__init__(args)

    def _build_ext_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-task", default='ext')
        parser.add_argument("-encoder", default='bert')
        parser.add_argument("-mode", default='train')
        parser.add_argument("-bert_model_path", default=src_path + '/../bert_pretrained/')
        parser.add_argument("-bert_data_path", default='./bert_data/cnndm')
        parser.add_argument("-model_path", default='./models/')
        parser.add_argument("-result_path", default='./results/cnndm')
        parser.add_argument('-log_file', default='./logs/cnndm.log')
        parser.add_argument("-temp_dir", default='./temp')
        parser.add_argument("-train_from", default='')

        parser.add_argument('-min_src_nsents', default=3, type=int)
        parser.add_argument('-max_src_nsents', default=100, type=int)
        parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
        parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)

        parser.add_argument("-max_pos", default=512, type=int)
        parser.add_argument("-max_tgt_len", default=140, type=int)
        parser.add_argument("-max_n_tags", default=6, type=int)
        parser.add_argument("-use_interval", default=True, type=bool)
        parser.add_argument("-large", default=False, type=bool)

        parser.add_argument("-ext_dropout", default=0.2, type=float)
        parser.add_argument("-ext_layers", default=2, type=int)
        parser.add_argument("-ext_hidden_size", default=768, type=int)
        parser.add_argument("-ext_heads", default=8, type=int)
        parser.add_argument("-ext_ff_size", default=2048, type=int)

        parser.add_argument("-param_init", default=0, type=float)
        parser.add_argument("-param_init_glorot", default=True, type=bool)
        parser.add_argument("-optim", default='adam')
        parser.add_argument("-lr", default=2e-3, type=float)
        parser.add_argument("-beta1", default=0.9, type=float)
        parser.add_argument("-beta2", default=0.999, type=float)
        parser.add_argument("-max_grad_norm", default=0, type=float)

        parser.add_argument("-train_steps", default=40000, type=int)
        parser.add_argument("-warmup_steps", default=10000, type=int)
        parser.add_argument("-report_every", default=50, type=int)
        parser.add_argument("-test_start_from", default=10000, type=int)
        parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
        parser.add_argument("-batch_size", default=8*512, type=int)
        parser.add_argument("-accum_count", default=2, type=int)
        parser.add_argument('-visible_gpus', default='0', type=str)
        parser.add_argument("-test_batch_size", default=8*512, type=int)

        parser.add_argument("-finetune_bert", default=True, type=bool)
        parser.add_argument('-gpu_ranks', default='0', type=str)
        parser.add_argument('-seed', default=666, type=int)

        parser.add_argument("-test_all", default=True, type=bool)
        parser.add_argument("-test_from", default='')
        parser.add_argument("-recall_eval", default=False, type=bool)
        parser.add_argument("-report_rouge", default=True, type=bool)
        parser.add_argument("-block_trigram", default=False, type=bool)

        args = parser.parse_args('')
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
        args.world_size = len(args.gpu_ranks)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
        args.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        args.device_id = 0 if args.device == "cuda" else -1
        return args

    def extract(self, batch):
        sel_scores, sel_ids = self.trainer.predict(batch)
        exts = self.trainer.generate_srcext(batch, sel_scores, sel_ids, self.decider)
        exts = [ext[0].split('<q>') for ext in exts]
        doc_exts = []
        for src_sents, src_idx, ext_sents in zip(batch.src_str, batch.src_idxs, exts):
            doc_exts.append([src_idx[src_sents.index(sent)] for sent in ext_sents])
        # doc_exts = [list(sorted(ext)) for ext in doc_exts]
        return doc_exts
