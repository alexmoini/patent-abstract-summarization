# SWE4S-DeepLearning-NLP-Project

## Software Usage
This software can be used to train transformers on masked language modeling and summarization tasks. Any model from the huggingface hub that is a LLM transformer can be used as long as it has an encoder-decoder architecture (encoder only CAN be used for MLM training exclusively).

### Training MLM

    pip install -r mlm/requirements.txt
    python mlm_train.py [--model-architecture MODEL_ARCHITECTURE] 
                       [--model-dir MODEL_DIR] 
                       [--train TRAIN] 
                       [--eval EVAL] 
                       [--output-dir OUTPUT_DIR] 
                       [--logs-dir LOGS_DIR] 
                       [--batch-size BATCH_SIZE]
                       [--num-epochs NUM_EPOCHS] 
                       [--learning-rate LEARNING_RATE]
                       [--num-workers NUM_WORKERS] 
                       [--gpus GPUS] 
                       [--hardware HARDWARE] 
                       [--max-seq-length MAX_SEQ_LENGTH] 
                       [--seed SEED]
                       [--masked-lm-prob MASKED_LM_PROB] 
                       [--pre-masked PRE_MASKED]

- model-architecture: HF Hub model architecture
- model-dir: pytorch state dict checkpoint file
- train: path to train data
- eval: path to eval data
- output-dir: path to desired output directory
- logs-dir: path to desired log directory
- batch-size: num samples per train batch
- num-epochs: number of epochs to run training for
- learning-rate: weight adjustment rate
- num-workers: cpu workers for loading data
- gpus: num gpus to use
- hardware: gpu type or cpu (mps, gpu, cpu)
- max-seq-length: length of input sequence desired max
- seed: randomization seeding for reproducibility
- masked-lm-prob: percentage of tokens to mask per input sequence
- pre-masked: if your data is premasked mark this true

#### EXPECTED DATA INPUT: 
Text file with samples separated by newline separator

#### EXAMPLE:
this is a sentence.\n this is another sentence for mlm. \n this is a third sample.

### Training Summarization

    pip install -r summarization/requirements.txt
    python summarization_train.py [--model-architecture MODEL_ARCHITECTURE]
                                  [--model-dir MODEL_DIR]
                                  [--train TRAIN]
                                  [--eval EVAL] 
                                  [--output-dir OUTPUT_DIR] 
                                  [--logs-dir LOGS_DIR] 
                                  [--batch-size BATCH_SIZE]
                                  [--num-epochs NUM_EPOCHS] 
                                  [--learning-rate LEARNING_RATE] 
                                  [--num-workers NUM_WORKERS] 
                                  [--gpus GPUS] 
                                  [--max-seq-length MAX_SEQ_LENGTH] 
                                  [--seed SEED]

- model-architecture: HF Hub model architecture
- model-dir: pytorch state dict checkpoint file
- train: path to train data
- eval: path to eval data
- output-dir: path to desired output directory
- logs-dir: path to desired log directory
- batch-size: num samples per train batch
- num-epochs: number of epochs to run training for
- learning-rate: weight adjustment rate
- num-workers: cpu workers for loading data
- gpus: num gpus to use
- hardware: gpu type or cpu (mps, gpu, cpu)
- max-seq-length: length of input sequence desired max
- seed: randomization seeding for reproducibility
- masked-lm-prob: percentage of tokens to mask per input sequence

#### EXPECTED DATA INPUT: 
CSV file with heading: {index}, description, abstract

#### EXAMPLE:
,description,abstract
0,"this is a desc", "this is an abstract"
1,"this is another desc", "this is another abs"


### Calling Summarization Model from HF Hub

    python summarization_api.py [--model-name MODEL_NAME]
                                 [--inference-hardware INFERENCE_HARDWARE]
                                 [--path-to-data PATH_TO_DATA]

- model-name: HF Hub model name
- inference-hardware: type of inference hardware to use that is available on your machine (gpu, mps or cpu compatible)
- path-to-data: path to data to transform

Recommended Model Names:
facebook/bart-base -> Will have decent performance on summarizing basic language
facebook/bart-large-cnn -> Will have good performance on news article summarization

alex-moini/bart-base-patent-summarization-no-pretrained-2022-11-5 -> Will have good performance on summarizing patents 
- (this was the model trained by these scripts)

#### EXPECTED DATA INPUT: 
JSON file with a single field, text, with the relevant text to summarize

#### EXAMPLE:
    {
      "text":"text to summarize"
    }

#### EXPECTED OUTPUT:
Will print summary and abstract as well as upload a file with a json like so:
    {
      "text": "text to summarize",
      "summary": "summarized text"
    }
### Using Python SDK to access your model
You must upload your model to HF hub repo to do this. Unfortunately because the repos require private access keys I cannot have the script uploading to a HF repo as it would no longer have a private id.

    !pip install transformers
    import transformers

    model_architecture = 'alex-moini/bart-base-patent-summarization-no-pretrained-2022-11-5'

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_architecture)

    text = 'Enter your favorite patent here!'

    #tokenize text
    tokens = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)

    #generate summary
    output = model.generate(tokens['input_ids'])

    #decode token ids to text
    text_out = tokenizer.batch_decode(output, skip_special_tokens=True)

    print(text_out)

## TODO List
2. functional tests mlm
3. functional tests summarization
4. test env yaml file and github actions
