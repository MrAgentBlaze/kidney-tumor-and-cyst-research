import torch
from torch.nn import functional as F


def ce_loss(inputs, targets, reduction="none", sigmoid=True):
    """Cross-entropy loss supporting both binary and multi-class modes."""
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(targets, list):
        targets = [targets]

    assert len(inputs) == len(targets), "Batch size mismatch between inputs and targets"
    batch_size = len(inputs)

    loss = torch.zeros(batch_size, device=inputs[0].device)
    if sigmoid:
        for idx, (ipt, tgt) in enumerate(zip(inputs, targets)):
            loss[idx] = F.binary_cross_entropy_with_logits(ipt, tgt, reduction=reduction)
    else:
        for idx, (ipt, tgt) in enumerate(zip(inputs, targets)):
            loss[idx] = F.cross_entropy(ipt, tgt, reduction=reduction)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def focal_loss(inputs, targets, alpha=None, gamma=0.0,
               reduction="none", sigmoid=True):
    """Focal loss for handling class imbalance."""
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(targets, list):
        targets = [targets]

    assert len(inputs) == len(targets), "Batch size mismatch between inputs and targets"
    batch_size = len(inputs)

    loss = torch.zeros(batch_size, device=inputs[0].device)
    if sigmoid:
        for idx, (ipt, tgt) in enumerate(zip(inputs, targets)):
            loss[idx] = _sigmoid_focal_loss(ipt, tgt, alpha, gamma, reduction)
    else:
        for idx, (ipt, tgt) in enumerate(zip(inputs, targets)):
            loss[idx] = _softmax_focal_loss(ipt, tgt, alpha, gamma, reduction)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def _softmax_focal_loss(inputs, targets, alpha=None, gamma=0.0, reduction="none"):
    """Multi-class focal loss using softmax."""
    if inputs.ndim > 2:
        c = inputs.shape[1]
        inputs = inputs.permute(0, *range(2, inputs.ndim), 1).reshape(-1, c)

    targets = targets.view(-1)
    inputs = inputs.float()
    targets = targets.long()

    log_p = F.log_softmax(inputs, dim=1)
    ce = F.nll_loss(log_p, targets, weight=alpha, reduction="none")

    all_rows = torch.arange(len(inputs))
    log_pt = log_p[all_rows, targets]
    pt = log_pt.exp()
    focal_term = (1 - pt) ** gamma
    loss = focal_term * ce

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def _sigmoid_focal_loss(inputs, targets, alpha=None, gamma=2.0, reduction="none"):
    """Binary focal loss using sigmoid (RetinaNet formulation)."""
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)

    if alpha:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def dice_score(inputs, targets, smooth_num=0, smooth_den=1e-12):
    """Compute Dice similarity coefficient."""
    reduce_axes = torch.arange(1, len(inputs.shape)).tolist()
    intersection = torch.sum(targets * inputs, dim=reduce_axes)
    union = torch.sum(targets, dim=reduce_axes) + torch.sum(inputs, dim=reduce_axes)
    return torch.mean((2.0 * intersection + smooth_num) / (union + smooth_den))


def dice_loss(inputs, targets, sigmoid=True,
              smooth_num=0, smooth_den=1e-12, reduction="none"):
    """Dice loss for segmentation tasks."""
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(targets, list):
        targets = [targets]

    assert len(inputs) == len(targets), "Batch size mismatch between inputs and targets"
    batch_size = len(inputs)

    scores = torch.zeros(batch_size, device=inputs[0].device)
    if sigmoid:
        for idx, (ipt, tgt) in enumerate(zip(inputs, targets)):
            if ipt.size(-1) == 1:
                ipt = ipt.squeeze(-1)
            if tgt.size(-1) == 1:
                tgt = tgt.squeeze(-1)
            ipt = torch.sigmoid(ipt)
            scores[idx] = dice_score(ipt, tgt, smooth_num, smooth_den)
    else:
        for idx, (ipt, tgt) in enumerate(zip(inputs, targets)):
            ipt = torch.softmax(ipt, 1)
            score, nb_labels = 0.0, ipt.size(1)
            for i in range(nb_labels):
                ipt_i = ipt[:, i]
                tgt_i = (tgt[:, 0] == i).float()
                score += dice_score(ipt_i, tgt_i, smooth_num, smooth_den)
            scores[idx] = score / nb_labels

    loss = 1 - scores
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss
