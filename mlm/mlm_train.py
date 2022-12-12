import sys
import io
from smart_open import open as smart_open
from mlm_tuner import MaskedLanguageModelingModel
from embedding_backbone import TextEmbeddingBackbone
from mlm_utils import CorpusMaskingDataset
import argparse
import random
import os
from torch.utils.data import DataLoader, Dataset
import logging
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import transformers
from transformers.utils import logging
from dagshub.pytorch_lightning import DAGsHubLogger
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--model-architecture", type=str,
            default="bert-base-uncased")
    parser.add_argument(
            "--model-dir", type=str,
            default=None)  # os.environ.get("SM_MODEL_DIR")
    parser.add_argument(
            "--train", type=str,
            default='s3://deeplearning-nlp-bucket/corpus-patent-data/train')
    parser.add_argument(
            "--eval", type=str,
            default='s3://deeplearning-nlp-bucket/corpus-patent-data/validation')  # nopep8
    parser.add_argument(
            "--output-dir", type=str,
            default='models/bart-large/')
    parser.add_argument(
            "--logs-dir", type=str,
            default='logs/bart-large/')
    parser.add_argument(
            "--batch-size",
            type=int, default=32)
    parser.add_argument(
            "--num-epochs",
            type=int, default=1)
    parser.add_argument(
            "--learning-rate",
            type=float, default=1e-3)
    parser.add_argument(
            "--num-workers",
            type=int, default=8)
    parser.add_argument(
            "--gpus",
            type=int, default=1)
    parser.add_argument(
            "--hardware",
            type=str, default="mps")
    parser.add_argument(
            "--max-seq-length",
            type=int, default=128)
    parser.add_argument(
            "--seed",
            type=int, default=42)
    parser.add_argument(
            "--masked-lm-prob",
            type=float, default=0.15)
    parser.add_argument(
            "--pre-masked",
            type=bool, default=False)
    args, _ = parser.parse_known_args()
    return args


def load_dataset(
        dataset_dir, tokenizer,
        pre_masked=False, sequence_length=128):
    dataset = CorpusMaskingDataset(
            dataset_dir, tokenizer,
            pre_masked=False,
            sequence_length=sequence_length)
    return dataset


def load_embedding_backbone(model_architecture, model_dir):
    embedding_backbone = TextEmbeddingBackbone(model_architecture,
                                               checkpoint_dir=model_dir,
                                               pool_outputs=False)
    return embedding_backbone


def load_mlm_model(embedding_backbone):
    mlm_model = MaskedLanguageModelingModel(embedding_backbone)
    return mlm_model


def directory_to_path(directory, file_name):
    path = os.path.join(directory, file_name)
    return path


def main():
    args = get_args()

    pl.seed_everything(args.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_architecture)
    embedding_backbone = load_embedding_backbone(
            args.model_architecture, args.model_dir)
    mlm_model = load_mlm_model(embedding_backbone)
    train_path = directory_to_path(args.train, 'train.csv')
    valid_path = directory_to_path(args.eval, 'validation.csv')
    train_dataset = load_dataset(
            train_path, tokenizer,
            pre_masked=args.pre_masked,
            sequence_length=args.max_seq_length)
    val_dataset = load_dataset(
            valid_path, tokenizer,
            pre_masked=args.pre_masked,
            sequence_length=args.max_seq_length)
    args.batch_size = args.batch_size * max(1, args.gpus)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    checkpoint_callback = pl.callbacks.ModelCheckpoint()
    earlystop_callback = pl.callbacks.EarlyStopping(
            monitor="train_loss",
            mode="min", patience=3,
            check_on_train_epoch_end=True)
    train_csv = train_path.split('/')[-1].split('.')[0]
    rand = random.randint(1, 100)
    version_ = (args.model_architecture
                + '-' + str(args.learning_rate)
                + '-' + str(args.batch_size)
                + '-' + train_csv
                + '-' + str(rand))
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.hardware == 'mps':
        trainer = pl.Trainer(devices=args.gpus,
                             accelerator='mps',
                             max_epochs=args.num_epochs,
                             logger=DAGsHubLogger(
                                 metrics_path=args.logs_dir+'metrics.csv',
                                 hparams_path=args.logs_dir+'hparams.csv'),
                             callbacks=[checkpoint_callback,
                                        earlystop_callback,
                                        lr_monitor]
                             )
    if args.gpus > 1:
        trainer = pl.Trainer(devices=args.gpus,
                             accelerator='gpu',
                             strategy='ddp',
                             max_epochs=args.num_epochs,
                             logger=DAGsHubLogger(
                                 metrics_path=args.logs_dir+'metrics.csv',
                                 hparams_path=args.logs_dir+'hparams.csv'),
                             callbacks=[checkpoint_callback,
                                        earlystop_callback,
                                        lr_monitor]
                             )
    if args.gpus == 0:
        raise ValueError("No GPUs found. Please set gpus > 0")
    # check if cuda is available
    if torch.cuda.is_available():
        print("CUDA being used")
    trainer = pl.Trainer(devices=args.gpus,
                         accelerator='gpu',
                         max_epochs=args.num_epochs,
                         logger=DAGsHubLogger(
                             metrics_path=args.logs_dir+'metrics.csv',
                             hparams_path=args.logs_dir+'hparams.csv'),
                         callbacks=[checkpoint_callback,
                                    earlystop_callback,
                                    lr_monitor]
                         )

    trainer.fit(mlm_model, train_dataloader, val_dataloader)
    best_model_path = checkpoint_callback.best_model_path
    with smart_open(best_model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    # buffer -> checkpoint_dir
    embedding_backbone = load_embedding_backbone(
            args.model_architecture, buffer)
    mlm_model = load_mlm_model(embedding_backbone)
    with open(os.path.join(args.output_dir, 'model.pth'), 'wb') as f:
        mlm_model.save_backbone(os.path.join(args.output_dir, 'model.pth'))


if __name__ == "__main__":
    main()
