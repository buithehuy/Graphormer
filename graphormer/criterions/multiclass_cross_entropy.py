# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass
from dataclasses import dataclass, field

import torch
from torch.nn import functional
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("multiclass_cross_entropy", dataclass=FairseqDataclass)
class GraphPredictionMulticlassCrossEntropy(FairseqCriterion):
    """
    Implementation for the multi-class log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])[: logits.size(0)]
        ncorrect = (torch.argmax(logits, dim=-1).reshape(-1) == targets.reshape(-1)).sum()

        loss = functional.cross_entropy(
            logits, targets.reshape(-1), reduction="sum"
        )

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": ncorrect,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("multiclass_cross_entropy_with_flag", dataclass=FairseqDataclass)
class GraphPredictionMulticlassCrossEntropyWithFlag(GraphPredictionMulticlassCrossEntropy):
    """
    Implementation for the multi-class log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]
        perturb = sample.get("perturb", None)

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"], perturb=perturb)
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])[: logits.size(0)]
        ncorrect = (torch.argmax(logits, dim=-1).reshape(-1) == targets.reshape(-1)).sum()

        loss = functional.cross_entropy(
            logits, targets.reshape(-1), reduction="sum"
        )

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": ncorrect,
        }
        return loss, sample_size, logging_output


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    use_class_weights: bool = field(
        default=False,
        metadata={"help": "use pre-calculated class weights for rice diseases dataset"},
    )
    use_focal_loss: bool = field(
        default=False,
        metadata={"help": "use focal loss"},
    )
    focal_gamma: float = field(
        default=2.0,
        metadata={"help": "gamma for focal loss"},
    )

@register_criterion("label_smoothed_cross_entropy_for_graph", dataclass=LabelSmoothedCrossEntropyCriterionConfig)
class GraphPredictionLabelSmoothedCrossEntropy(FairseqCriterion):
    """
    Implementation for the label smoothed multi-class log loss used in graphormer model training.
    """
    def __init__(self, task, label_smoothing, use_class_weights=False, use_focal_loss=False, focal_gamma=2.0):
        super().__init__(task)
        self.eps = label_smoothing
        self.use_class_weights = use_class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        self.class_weights = torch.tensor([1.60, 0.56, 1.48, 1.07], dtype=torch.float32)

    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])[: logits.size(0)]
        ncorrect = (torch.argmax(logits, dim=-1).reshape(-1) == targets.reshape(-1)).sum()

        lprobs = functional.log_softmax(logits, dim=-1, dtype=torch.float32)
        targets = targets.reshape(-1)
        
        weight = self.class_weights.to(logits.device) if self.use_class_weights else None
        
        if self.use_focal_loss:
            pt = torch.exp(lprobs.gather(1, targets.unsqueeze(1)))
            focal_weight = (1 - pt) ** self.focal_gamma
            
            nll_loss_none = functional.nll_loss(lprobs, targets, weight=weight, reduction="none")
            nll_loss = (nll_loss_none * focal_weight.squeeze(-1)).sum()
            
            if self.use_class_weights:
                smooth_loss_none = - (lprobs * weight.unsqueeze(0)).sum(dim=-1, keepdim=True).squeeze(-1)
                smooth_loss = (smooth_loss_none * focal_weight.squeeze(-1)).sum()
            else:
                smooth_loss_none = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
                smooth_loss = (smooth_loss_none * focal_weight.squeeze(-1)).sum()
                
            loss = (1.0 - self.eps) * nll_loss + (self.eps / logits.size(-1)) * smooth_loss
        else:
            nll_loss = functional.nll_loss(lprobs, targets, weight=weight, reduction="sum")
            if self.use_class_weights:
                smooth_loss = - (lprobs * weight.unsqueeze(0)).sum(dim=-1, keepdim=True).sum()
            else:
                smooth_loss = -lprobs.sum(dim=-1, keepdim=True).sum()
                
            loss = (1.0 - self.eps) * nll_loss + (self.eps / logits.size(-1)) * smooth_loss

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": ncorrect,
        }
        
        # Calculate TP, FP, FN for Macro F1
        preds = torch.argmax(logits, dim=-1).reshape(-1)
        num_classes = logits.size(-1)
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum()
            fp = ((preds == c) & (targets != c)).sum()
            fn = ((preds != c) & (targets == c)).sum()
            logging_output[f"tp_c{c}"] = tp.data
            logging_output[f"fp_c{c}"] = fp.data
            logging_output[f"fn_c{c}"] = fn.data

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("nll_loss", nll_loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=1
            )
            
            # Calculate Macro F1 and Recall
            num_classes = 4 # Hardcode based on rice diseases dataset
            macro_recall = 0.0
            macro_f1 = 0.0
            valid_classes = 0
            
            for c in range(num_classes):
                tp = sum(log.get(f"tp_c{c}", 0) for log in logging_outputs)
                fp = sum(log.get(f"fp_c{c}", 0) for log in logging_outputs)
                fn = sum(log.get(f"fn_c{c}", 0) for log in logging_outputs)
                
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                    macro_recall += recall
                
                if tp + fp + fn > 0:
                    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
                    macro_f1 += f1
                    valid_classes += 1
            
            if valid_classes > 0:
                metrics.log_scalar("macro_recall", 100.0 * macro_recall / valid_classes, sample_size, round=1)
                metrics.log_scalar("macro_f1", 100.0 * macro_f1 / valid_classes, sample_size, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
