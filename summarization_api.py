import transformers
import torch
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='facebook/bart-base')
    parser.add_argument('--inference-hardware', type=str, default='cpu')
    parser.add_argument('--path-to-data', type=str, default=None)
    args = parser.parse_args()
    return args

def load_data():
    """
    Load data from json file
    """
    args = get_args()
    # read json file using with open
    with open(args.path_to_data, 'r') as f:
        data = json.load(f)
    return data

def shorten_text(text, max_length):
    """
    Shorten text to max_length
    """
    if len(text) > max_length:
        text = text.split(' ')[:int(max_length*.75)]
    return text

def get_model(model_name, inference_hardware):
    """
    Load model and tokenizer
    """
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if inference_hardware == 'gpu':
        model = model.cuda()
    elif inference_hardware == 'mps':
        model = model.to('cpu')
    model.eval()
    return model, tokenizer

def predict(text):
    """
    Predict summary
    """
    args = get_args()
    model, tokenizer = get_model(args.model_name, args.inference_hardware)
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    if args.inference_hardware == 'gpu':
        inputs = inputs.cuda()
    elif args.inference_hardware == 'mps':
        inputs = inputs.to('cpu')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    data = load_data()
    data['summary'] = predict(data['text'])
    print(data)
    # write to json file
    with open('summarization_output.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()