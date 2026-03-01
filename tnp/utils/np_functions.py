import torch
from torch import nn

from ..data.base import Batch, ImageBatch
from ..models.base import (
    ARConditionalNeuralProcess,
    ConditionalNeuralProcess,
    LatentNeuralProcess,
    CausalNeuralProcess,
)
from ..models.convcnp import GriddedConvCNP


def np_pred_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.distributions.Distribution:
    if isinstance(model, GriddedConvCNP):
        assert isinstance(batch, ImageBatch)
        pred_dist = model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
    elif isinstance(model, ConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
    elif isinstance(model, LatentNeuralProcess):
        pred_dist = model(
            xc=batch.xc, yc=batch.yc, xt=batch.xt, num_samples=num_samples
        )
    elif isinstance(model, ARConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt)
    elif isinstance(model, CausalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt)
    else:
        raise ValueError

    return pred_dist


def np_loss_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.Tensor:
    """Perform a single training step, returning the loss, i.e.
    the negative log likelihood.

    Arguments:
        model: model to train.
        batch: batch of data.

    Returns:
        loss: average negative log likelihood.
    """
    pred_dist = np_pred_fn(model, batch, num_samples)
    loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()

    return -loglik


def full_sequence_loss_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Calculates NLL over the entire sequence (context + target).
    """
    # 1. Get predictions (Nc + Nt points if using FullSequenceDecoder)
    pred_dist = np_pred_fn(model, batch, num_samples)
    
    # 2. Concatenate context and target ground truths
    y_all = torch.cat([batch.yc, batch.yt], dim=1)
    
    # 3. Calculate log-likelihood against the full sequence
    loglik = pred_dist.log_prob(y_all).sum() / y_all[..., 0].numel()

    return -loglik