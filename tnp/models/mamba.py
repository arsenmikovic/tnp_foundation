from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn
import copy

from ..networks.transformer import ISTEncoder, PerceiverEncoder, TNPTransformerEncoder
from ..networks.mamba import MNPNDMambaEncoder, TNPMambaEncoder
from .base import CausalNeuralProcess
from tnp.utils.helpers import preprocess_observations


class FullSequenceDecoder(nn.Module):
    """
    Decodes the entire sequence of representations (z) without slicing.
    Used for causal/temporal models where loss is calculated on all points.
    """
    def __init__(self, z_decoder: nn.Module):
        super().__init__()
        self.z_decoder = z_decoder

    @check_shapes("z: [m, ..., n, dz]", "return: [m, ..., n, dy]")
    def forward(
        self, z: torch.Tensor, xt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # We ignore xt because we want the model to decode EVERY point 
        # provided by the encoder (both context and target).
        return self.z_decoder(z)



class CausalTemporalMambaEncoder(nn.Module):
    def __init__(
        self, 
        mamba_layer: nn.Module, 
        xy_encoder: nn.Module,  # Add this
        num_layers: int, 
    ):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(mamba_layer) for _ in range(num_layers)])
        self.xy_encoder = xy_encoder # Use this instead of input_projection

    def forward(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
        #yc, yt = preprocess_observations(xt, yc)
        full_x = torch.cat([xc, xt], dim=1)  
        full_y_0 = torch.cat([yc, yt], dim=1) 

        full_y, _ = preprocess_observations(xt, full_y_0) 

        b, n, _ = full_y.shape
        zeros = torch.zeros(b, 1, full_y.shape[-1], device=full_y.device)
        shifted_y = torch.cat([zeros, full_y[:, :-1, :]], dim=1) 

        tokens = torch.cat([full_x, shifted_y], dim=-1) 
        
        # Use the MLP xy_encoder here!
        z = self.xy_encoder(tokens) 

        for layer in self.layers:
            z = layer(z)
        return z


class MAMBA(CausalNeuralProcess):
    def __init__(
        self,
        encoder: CausalTemporalMambaEncoder,
        decoder: FullSequenceDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


