import pytorch_lightning as pl
import torch
import transformers
import pandas as pd
import torchmetrics as tm
from transformers.utils import logging
from torchmetrics.text.rouge import ROUGEScore

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class SummarizationModel(pl.LightningModule):
    """

    """
    def __init__(self, model_architecture, checkpoint_dir=None, learning_rate=1e-4, loss=torch.nn.CrossEntropyLoss()):
        """

        """
        super().__init__()
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_architecture)
        if checkpoint_dir is not None:
            self.model.load_state_dict(torch.load(checkpoint_dir), strict=False)
        self.learning_rate = learning_rate
        self.loss = loss
        self.rouge = ROUGEScore()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_architecture)

    def forward(self, input_ids, attention_masks, labels):
        """
        Forward pass of the model
        """
        return self.model(input_ids, attention_masks, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Training step of the model
        """
        torch.cuda.empty_cache()
        self.model.train()
        encoding = batch
        model_output = self(encoding['input_ids'], encoding['attention_mask'], encoding['labels'])
        loss = model_output.loss
        output_tokens = model_output.logits.argmax(dim=2)
        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        labels_text = self.tokenizer.batch_decode(encoding['labels'], skip_special_tokens=True)
        scores = self.rouge(output_text, labels_text)
        self.log("train_loss", loss)
        self.log("train_rouge1", scores['rouge1_fmeasure'])
        self.log("train_rouge2", scores['rouge2_fmeasure'])
        self.log("train_rougeL", scores['rougeL_fmeasure'])
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model
        """
        torch.cuda.empty_cache()
        self.model.eval()
        encoding = batch
        with torch.no_grad():
            model_output = self(encoding['input_ids'], encoding['attention_mask'], encoding['labels'])
        loss = model_output.loss
        output_tokens = model_output.logits.argmax(dim=2)
        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        labels_text = self.tokenizer.batch_decode(encoding['labels'], skip_special_tokens=True)
        scores = self.rouge(output_text, labels_text)
        self.log("train_loss", loss)
        self.log("train_rouge1", scores['rouge1_fmeasure'])
        self.log("train_rouge2", scores['rouge2_fmeasure'])
        self.log("train_rougeL", scores['rougeL_fmeasure'])
        return loss

    def predict(self, text, max_length=200, beams=2, early_stopping=True):
        """
        Predict the summary of the text
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=max_length, num_beams=beams, early_stopping=early_stopping)
        return self.tokenizer.batch_decode(output[0], skip_special_tokens=True)

    def configure_optimizers(self):
        """
        Configure the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "monitor": "train_loss"
        }
        }

    def save_model(self, save_dir):
        """
        Save the model
        """
        torch.save(self.model.state_dict(), save_dir)
