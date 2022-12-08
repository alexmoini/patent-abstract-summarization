import datasets
from nltk import sent_tokenize
import boto3
import time
import argparse
import nltk
import pandas as pd
nltk.download('punkt')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--output_dir', type=str, default='s3://deeplearning-nlp-bucket/summary_patent_data/')
    parser.add_argument('--dataset_name', type=str, default='big_patent')
    parser.add_argument('--dataset_config_name', type=str, default='d')
    parser.add_argument('--test_data', type=bool, default=False)
    args = parser.parse_args()
    return args

def load_and_split(args):
    print("Loading dataset")
    start = time.time()
    dataset = datasets.load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    end = time.time()
    print("Loaded dataset in {} seconds".format(end-start))
    dataset = dataset.to_pandas()
    # only use BACK half of dataset if train
    if args.split == 'train':
        dataset = dataset[int(len(dataset)/2):]
    if args.test_data:
        dataset = dataset[:5]
        args.split = 'unittest_data'
    # upload dataset as csv
    if args.output_dir.startswith('s3://'):
        s3 = boto3.client('s3')
        bucket, key = args.output_dir[5:].split('/', 1)
        s3.put_object(Bucket=bucket, Key=key+args.split, Body=dataset.to_csv())
    else:
        dataset.to_csv(args.output_dir+args.split)

if __name__ == '__main__':
    args = get_args()
    load_and_split(args)