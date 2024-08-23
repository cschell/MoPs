import lightning as L
import torch
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import CustomKNN

from src.models.rnn import RNN
from src.models.rnn_transformer import RNNTransformer


class SimilarityLearning(L.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.arch == "RNN":
            self.model = RNN(**kwargs, cell_type="GRU", keras_initialization=False)
        elif self.hparams.arch == "RNNTransformer":
            self.model = RNNTransformer(**kwargs)

        self.loss_func = ArcFaceLoss(
            num_classes=self.hparams.n_train_classes,
            embedding_size=self.hparams.embedding_size,
            margin=self.hparams.arc_face_loss_margin,
            scale=self.hparams.arc_face_loss_scale,
            weight_reg_weight=self.hparams.arc_face_loss_weight_reg_weight,
        )

        self.batch_tuning_mode = False
        self.scaling_params = kwargs.get("scaling_params", None)

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        if isinstance(batch[0], list):
            batches = batch
        else:
            batches = [batch]

        losses = []
        for batch in batches:
            X, y, *lengths = batch
            if lengths:
                random_lengths = lengths[0]
            else:
                seq_length = X.shape[1]
                random_lengths = torch.randint(round(seq_length * 0.5), seq_length + 1, (len(X),))

            embeddings = self.forward(
                X,
                random_lengths,
            )

            loss = self.loss_func(embeddings, y)
            losses.append(loss)

        if len(losses) > 1:
            loss = (losses[0] + losses[1]) / 2
        else:
            loss = losses[0]
        self.log(f"loss/train", loss, on_step=False, on_epoch=True, batch_size=len(X), prog_bar=True)

        return loss

    def on_validation_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, batch, *args):
        X, y = batch

        h = self.forward(X)

        self.validation_step_outputs.append((h, y))

    def on_validation_epoch_end(self):
        if self.batch_tuning_mode:
            return

        wia_embeddings = torch.cat([emb for emb, _ in self.validation_step_outputs]).cpu()
        wia_y = torch.cat([y for _, y in self.validation_step_outputs]).cpu()

        calc = AccuracyCalculator(
            include=["precision_at_1", "r_precision"],
            k="max_bin_count",
            device=torch.device("cpu"),
            knn_func=CustomKNN(self.loss_func.distance),
        )
        wia_metrics = calc.get_accuracy(wia_embeddings, wia_y, ref_includes_query=True)
        self.log(f"precision_at_1/val", wia_metrics["precision_at_1"], prog_bar=True)
        self.log(f"r_precision/val", wia_metrics["r_precision"], prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=self.hparams.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["scaling_params"] = self.scaling_params

    def on_load_checkpoint(self, checkpoint):
        self.scaling_params = checkpoint.get("scaling_params")

    def on_train_start(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if not self.trainer.fast_dev_run:
            self.logger.experiment.summary["total_trainable_params"] = total_params
