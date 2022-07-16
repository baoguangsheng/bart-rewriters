# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, tag_range=None, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    accuracy = (lprobs.argmax(dim=-1, keepdim=True) == target).int()
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        accuracy.masked_fill_(pad_mask, 0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    # Guangsheng Bao: log the loss on special tokens for group tag
    if tag_range is not None:
        tag_mask = (target >= tag_range[0]).float() * (target <= tag_range[1]).float()
        ntokens_tag = tag_mask.sum(dim=-1)
        nll_loss_tag = nll_loss * tag_mask
        nll_loss_tok = nll_loss * (1 - tag_mask)
        accuracy_tag = accuracy * tag_mask
        accuracy_tok = accuracy * (1 - tag_mask)
    else:
        ntokens_tag = torch.zeros_like(target)
        nll_loss_tag = torch.zeros_like(nll_loss)
        nll_loss_tok = torch.zeros_like(nll_loss)
        accuracy_tag = torch.zeros_like(accuracy)
        accuracy_tok = torch.zeros_like(accuracy)

    if reduce:
        ntokens_tag = ntokens_tag.sum()
        nll_loss_tag = nll_loss_tag.sum()
        nll_loss_tok = nll_loss_tok.sum()
        # nll_loss = nll_loss.sum()
        nll_loss = nll_loss_tok * 0.8 + nll_loss_tag
        smooth_loss = smooth_loss.sum()
        accuracy_tag = accuracy_tag.sum()
        accuracy_tok = accuracy_tok.sum()

    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss_tok, nll_loss_tag, accuracy_tok, accuracy_tag, ntokens_tag

@register_criterion("label_smoothed_cross_entropy_rewriter")
class LabelSmoothedCrossEntropyRewriterCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss_tok, nll_loss_tag, accuracy_tok, accuracy_tag, ntokens_tag = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            'nll_loss_tok': nll_loss_tok.data,
            'nll_loss_tag': nll_loss_tag.data,
            'accuracy_tok': accuracy_tok.data,
            'accuracy_tag': accuracy_tag.data,
            'ntokens_tag': ntokens_tag.data,
            'ntokens': sample['ntokens'],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        tag_range = (self.task.target_dictionary.index('<S>'),
                     self.task.target_dictionary.index('</S>'))
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        return label_smoothed_nll_loss(
            lprobs, target, self.eps, tag_range=tag_range, ignore_index=self.padding_idx, reduce=reduce,
        )

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss_tok"].avg))

        # Guangsheng Bao: log loss on word tokens and special tokens for group tag
        nll_loss_tok_sum = sum(log.get('nll_loss_tok', 0).sum() for log in logging_outputs)
        nll_loss_tag_sum = sum(log.get('nll_loss_tag', 0).sum() for log in logging_outputs)
        accuracy_tok_sum = sum(log.get('accuracy_tok', 0).sum() for log in logging_outputs)
        accuracy_tag_sum = sum(log.get('accuracy_tag', 0).sum() for log in logging_outputs)
        ntokens_tag = sum(log.get('ntokens_tag', 0).sum() for log in logging_outputs)

        metrics.log_scalar('nll_loss_tok', nll_loss_tok_sum / (ntokens - ntokens_tag) / math.log(2), (ntokens - ntokens_tag), round=3)
        metrics.log_scalar('nll_loss_tag', nll_loss_tag_sum / ntokens_tag / math.log(2), ntokens_tag, round=3)
        metrics.log_scalar('accuracy_tok', accuracy_tok_sum / (ntokens - ntokens_tag), (ntokens - ntokens_tag), round=3)
        metrics.log_scalar('accuracy_tag', accuracy_tag_sum / ntokens_tag, ntokens_tag, round=3)

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
