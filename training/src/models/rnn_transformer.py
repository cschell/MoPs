import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNTransformer(nn.Module):

    def __init__(
        self,
        num_features: int,
        embedding_size: int,
        rnn_type: str,
        rnn_dropout: float,
        num_rnn_layers: int,
        d_model: int,
        dim_feedforward: int,
        dropout_frames: float,
        dropout_global: float,
        nhead: int,
        num_layers: int,
        **_kwargs,
    ):
        super().__init__()
        self.hparams = dict(
            embedding_size=embedding_size,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout_global=dropout_global,
            nhead=nhead,
            num_layers=num_layers,
            rnn_type=rnn_type,
            num_rnn_layers=num_rnn_layers,
            rnn_dropout=rnn_dropout,
            dropout_frames=dropout_frames,
        )

        self.embedding_size = embedding_size

        self.pos_encoder = nn.Identity()

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dim_feedforward=dim_feedforward,
                dropout=float(dropout_global),
            ),
            num_layers=num_layers,
        )

        self.frame_dropout = nn.Dropout(p=dropout_frames)

        rnn_class = getattr(nn, self.hparams["rnn_type"])
        self.projection_layer = rnn_class(
            input_size=num_features,
            hidden_size=d_model,
            dropout=float(self.hparams["rnn_dropout"]),
            num_layers=self.hparams["num_rnn_layers"],
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(float(dropout_global)),
            nn.ReLU(),
            nn.Linear(d_model, embedding_size),
        )

    def forward(self, x, lengths=None):
        x = self.frame_dropout(x)
        rnn_output, _ = self.projection_layer(x)
        x = self.pos_encoder(rnn_output)

        if lengths is not None:
            lengths = lengths.cpu().to(torch.int64)
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None].to(x.device)
        else:
            mask = torch.ones(x.shape[:-1], device=x.device).bool()

        x = self.transformer_encoder(x, src_key_padding_mask=~mask)

        if lengths is not None:
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1).to(x.dtype).to(x.device)
        else:
            x = x.mean(dim=1)

        x = self.output_layer(x)

        return x
