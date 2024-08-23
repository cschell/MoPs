from torch import nn
import torch


class RNN(nn.Module):

    def __init__(
        self,
        num_features: int,
        embedding_size: int,
        num_rnn_layers: int,
        rnn_hidden_size: int,
        dropout: float,
        cell_type: str,
        keras_initialization: bool = False,
        dropout_frames: float = 0.0,
        frame_noise: float = 0.0,
        **_kwargs
    ):
        super().__init__()

        self.num_features = num_features
        self.embedding_size = embedding_size
        self.hparams = dict(
            num_rnn_layers=num_rnn_layers,
            rnn_hidden_size=rnn_hidden_size,
            dropout=dropout,
            cell_type=cell_type,
            keras_initialization=keras_initialization,
            dropout_frames=dropout_frames,
            frame_noise=frame_noise,
        )
        self.dropout_frames = nn.Dropout(p=dropout_frames)
        self.rnn = getattr(nn, self.hparams["cell_type"])(
            input_size=self.num_features,
            hidden_size=self.hparams["rnn_hidden_size"],
            batch_first=True,
            dropout=self.hparams["dropout"],
            num_layers=self.hparams["num_rnn_layers"],
        )
        self.output_layer = nn.Linear(self.hparams["rnn_hidden_size"], self.embedding_size)

        if self.hparams["keras_initialization"]:
            # Applying Glorot (Xavier) Initialization to match Keras
            for name, param in self.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
                else:
                    nn.init.xavier_uniform_(param.data)

    def forward(self, x):

        sequences_end = (x == 0).all(axis=-1).sum()

        if self.hparams["frame_noise"]:
            x += torch.randn_like(x) * self.hparams["frame_noise"]
        x = self.dropout_frames(x)
        x, _rnn_state = self.rnn(x)
        # x = x[:, -1]
        x = x[:, -sequences_end]
        x = self.output_layer(x)

        return x
