import datasets
from nltk import sent_tokenize
import boto3
import time
import argparse
import nltk
nltk.download('punkt')
# get S3 role

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--output_dir', type=str, default='s3://deeplearning-nlp-bucket/corpus-patent-data/')
    parser.add_argument('--dataset_name', type=str, default='big_patent')
    parser.add_argument('--dataset_config_name', type=str, default='d')
    parser.add_argument('--text_column', type=str, default='description')
    parser.add_argument('--test_data', type=bool, default=False)
    args = parser.parse_args()
    return args

def load_and_split(args):
    print("Loading dataset")
    start = time.time()
    dataset = datasets.load_dataset('big_patent', 'd', split=args.split)
    end = time.time()
    print("Loaded dataset in {} seconds".format(end-start))
    dataset = dataset.to_pandas()
    # only use front half of dataset if train
    if args.split == 'train':
        dataset = dataset[:int(len(dataset)/2)]
    mlm_set = []
    for sample in dataset['description']:
        sentences = sent_tokenize(sample)
        for sentence in sentences:
            if len(sentence) > 50:
                mlm_set.append(sentence)
    # convert mlm_set to csv with \n delim
    mlm_set = '\n'.join(mlm_set)
    
    if args.test_data:
        mlm_set = '\n'.join(mlm_set[:5])
        args.split = 'unittest_data'
    # write to S3
    if args.output_dir.startswith('s3://'):
        s3 = boto3.client('s3')
        bucket, key = args.output_dir[5:].split('/', 1)
        s3.put_object(Bucket=bucket, Key=key+args.split, Body=mlm_set)
    else:
        with open(args.output_dir+args.split, 'w') as f:
            f.write(mlm_set)
if __name__ == '__main__':
    args = get_args()
    load_and_split(args.split)
