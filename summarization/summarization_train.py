from summarization_utils import SummarizationDataModule
from summarization_tuner import SummarizationModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import argparse
import os
import random
import torch
import logging
import transformers
from transformers.utils import logging
from dagshub.pytorch_lightning import DAGsHubLogger

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-architecture", type=str, default="facebook/bart-large")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--train", type=str, default='s3://deeplearning-nlp-bucket/summary_patent_data/train')
    parser.add_argument("--eval", type=str, default='s3://deeplearning-nlp-bucket/summary_patent_data/validation')
    parser.add_argument("--output-dir", type=str, default='/Users/alexandermoini/SWE4S-DeepLearning-NLP-Project/summarization/models/bart-base')
    parser.add_argument("--logs-dir", type=str, default='/Users/alexandermoini/SWE4S-DeepLearning-NLP-Project/summarization/logs/bart-base')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pre-masked", type=bool, default=False)
    args, _ = parser.parse_known_args()
    return args

def load_dataloaders(dataset_path_train, dataset_path_eval, tokenizer, sequence_length, batch_size, num_workers):
    train_dataset = SummarizationDataModule(dataset_path_train, tokenizer, max_length=sequence_length)
    val_dataset = SummarizationDataModule(dataset_path_eval, tokenizer, max_length=sequence_length)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_dataloader, val_dataloader

def load_summarization_model(model_architecture, checkpoint_dir, learning_rate):
    model = SummarizationModel(model_architecture, checkpoint_dir, learning_rate)
    return model

def train(args):
    pl.seed_everything(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_architecture)
    logger.info("Loading dataloaders")
    train_dataloader, val_dataloader = load_dataloaders(args.train, args.eval, tokenizer, args.max_seq_length, args.batch_size, args.num_workers)
    logger.info("Loading model")
    model = load_summarization_model(args.model_architecture, args.model_dir, args.learning_rate)
    logger.info("Creating callbacks")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.output_dir,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger.info("Creating trainer")
    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='cpu',
        precision=16,
        logger=DAGsHubLogger(metrics_path=args.logs_dir+'metrics.csv', hparams_path=args.logs_dir+'hparams.csv'),
    )
    logger.info("Starting training")
    trainer.fit(model, train_dataloader, val_dataloader)
    logger.info("Training complete")
    # save the model
    model.save_model(args.output_dir)

if __name__ == "__main__":
    args = get_args()
    train(args)