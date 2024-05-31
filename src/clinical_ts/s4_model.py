__all__ = ['S4Model']
#adapted from https://github.com/HazyResearch/state-spaces/blob/main/example.py
import torch
import torch.nn as nn

from clinical_ts.s42 import S4 as S42


class S4Model(nn.Module):

    def __init__(
        self, 
        d_input, # None to disable encoder
        d_output, # None to disable decoder
        d_state=64, #MODIFIED: N
        d_model=512, #MODIFIED: H
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
        l_max=1024,
        transposed_input=True, # behaves like 1d CNN if True else like a RNN with batch_first=True
        bidirectional=True, #MODIFIED
        layer_norm = True, # MODIFIED
        pooling = True, # MODIFIED
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.transposed_input = transposed_input
        
        # MODIFIED TO ALLOW FOR MODELS WITHOUT ENCODER
        if(d_input is None):
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Conv1d(d_input, d_model, 1) if transposed_input else nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S42(
                    d_state=d_state,
                    l_max=l_max,
                    d_model=d_model, 
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout, 
                    transposed=True,
                ))
            #MODIFIED TO ALLOW BATCH NORM MODELS
            self.layer_norm = layer_norm
            if(layer_norm):
                self.norms.append(nn.LayerNorm(d_model))
            else: #MODIFIED
                self.norms.append(nn.BatchNorm1d(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        self.pooling = pooling
        # Linear decoder
        # MODIFIED TO ALLOW FOR MODELS WITHOUT DECODER
        if(d_output is None):
            self.decoder = None
        else:
            self.decoder = nn.Linear(d_model, d_output)

    #MODIFIED
    def forward(self, x, rate=1.0):
        """
        Input x is shape (B, d_input, L) if transposed_input else (B, L, d_input)
        """
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L) if transposed_input else (B, L, d_input) -> (B, L, d_model)

        if(self.transposed_input is False):
            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                # MODIFIED
                z = norm(z.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)
            
            # Apply S4 block: we ignore the state input and output
            # MODIFIED
            z, _ = layer(z, rate=rate)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                # MODIFIED
                x = norm(x.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)

        x = x.transpose(-1, -2) # (B, d_model, L) -> (B, L, d_model)

        # MODIFIED ALLOW TO DISABLE POOLING
        if(self.pooling):
            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)

        # Decode the outputs
        if(self.decoder is not None):
            x = self.decoder(x)  # (B, d_model) -> (B, d_output) if pooling else (B, L, d_model) -> (B, L, d_output)
            
        if(not self.pooling and self.transposed_input is True):
            x = x.transpose(-1, -2) # (B, L, d_output) -> (B, d_output, L)
        return x
