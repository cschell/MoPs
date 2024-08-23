from torch import nn


class BaseTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        window_size: int,
        num_out_classes: int,
        d_model: int,
        dim_feedforward: int,
        pe_dropout: float,
        dropout_global: float,
        nhead: int,
        num_layers: int,
        positional_encoding: str,
    ):
        super().__init__()

        self.hparams = dict(
            window_size=window_size,
            num_out_classes=num_out_classes,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            pe_dropout=pe_dropout,
            dropout_global=dropout_global,
            nhead=nhead,
            num_layers=num_layers,
            positional_encoding=positional_encoding,
        )

        self.num_out_classes = num_out_classes

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

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)

        return x
