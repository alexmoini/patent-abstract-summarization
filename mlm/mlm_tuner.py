import pytorch_lightning as pl
import torch
import transformers
import pandas as pd
import torchmetrics as tm
from transformers.utils import logging
import logging


class MaskedLanguageModelingModel(pl.LightningModule):
    """
    Masked language modeling model
    """
    def __init__(
            self, embedding_backbone, learning_rate=1e-4,
            loss=torch.nn.CrossEntropyLoss()):
        """
        Model name can be any of the models from the transformers library
        Freeze is a boolean to freeze the weights of the model
        Pool outputs is a boolean to pool the outputs of the model
        Learning rate is the learning rate for the optimizer
        """
        super().__init__()
        self.model = embedding_backbone
        self.fc = torch.nn.Linear(self.model.config.hidden_size,
                                  self.model.config.vocab_size,
                                  bias=False)
        self.bias = torch.nn.Parameter(
                torch.zeros(self.model.config.vocab_size))

        # Need a link between the two variables so that the bias
        # is correctly resized with `resize_token_embeddings`
        self.fc.bias = self.bias
        self.accuracy = tm.Accuracy(
                task="multiclass",
                num_classes=self.model.config.vocab_size)
        self.loss = loss
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model
        """
        embeddings = self.model(input_ids, attention_mask)
        pred = self.fc(embeddings)
        return pred

    def training_step(self, batch, batch_idx):
        """
        Training step of the model
        """
        torch.cuda.empty_cache()
        masked_sequence, groundtruth_sequence = batch
        model_output = self(
                masked_sequence['input_ids'],
                masked_sequence['attention_mask'])
        loss = self.loss(
                model_output.view(-1, self.model.config.vocab_size),
                groundtruth_sequence['input_ids'].view(-1))
        acc = self.accuracy(
                model_output.view(-1, self.model.config.vocab_size),
                groundtruth_sequence['input_ids'].view(-1))
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        # Load to cloudwatch for hyperparameter tuning (monitoring train loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model
        """
        torch.cuda.empty_cache()
        masked_sequence, groundtruth_sequence = batch
        with torch.no_grad():
            model_output = self(
                    masked_sequence['input_ids'],
                    masked_sequence['attention_mask'])
        loss = self.loss(
                model_output.view(-1, self.model.config.vocab_size),
                groundtruth_sequence['input_ids'].view(-1))
        acc = self.accuracy(
                model_output.view(-1, self.model.config.vocab_size),
                groundtruth_sequence['input_ids'].view(-1))
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        # Load to cloudwatch for hyperparameter tuning (monitoring val loss)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                # If "monitor" references validation metrics,
                # then "frequency" should be set to a multiple of
                # "trainer.check_val_every_n_epoch".
                "monitor": "train_loss"
            }
        }

    def save_backbone(self, save_dir):
        """
        Save backbone embedding model
        """
        torch.save(self.model.state_dict(), save_dir)
