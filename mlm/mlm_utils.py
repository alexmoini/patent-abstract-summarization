from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import time
import copy

class CorpusMaskingDataset(Dataset):
    """
    Dataset to prepare and load a corpus of patents
    """
    def __init__(self, path, tokenizer, pre_masked=False, mask_prob=0.15, sequence_length=512, seed=42):
        """
        Path is the path to the csv file containing the corpus
        Tokenizer is the tokenizer to use to tokenize the corpus
        """
        if pre_masked:
            self.data = pd.read_csv(path, delimiter='\n')
        else:
            self.data = pd.read_csv(path, header=None, delimiter='\r\n', encoding='utf-8')
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.pre_masked = pre_masked
        self.sequence_length = sequence_length
        self.seed = seed

    def __random_masking__(self, masked_tokens, mask_prob):
        """
        Randomly masks the tokens with the mask probability
        """
        np.random.seed(self.seed)
        for i in range(1, len(masked_tokens)):
            random_number = np.random.rand()
            if masked_tokens[i] == 102:
                # WANT TO STOP MASKING AT END OF INPUT SEQUENCE
                # 102 INDICATES END OF INPUT SEQUENCE
                break
            if random_number <= mask_prob:
                masked_tokens[i] = 103
        return masked_tokens

    def __getitem__(self, index):
        """
        Returns a dictionary containing the input ids, attention mask and token type ids
        """
        if self.pre_masked: # If the data is already masked
            groundtruth_sequence = self.data.iloc[index]['groundtruth_sequence']
            masked_sequence = self.data.iloc[index]['masked_sequence']

            masked_sequence_tokens = self.tokenizer(masked_sequence, padding='max_length', max_length=self.sequence_length,
                                                    truncation=True, return_tensors="pt")
            groundtruth_sequence_tokens = self.tokenizer(groundtruth_sequence, padding='max_length', max_length=self.sequence_length,
                                                            truncation=True, return_tensors="pt")
            for key in masked_sequence_tokens.keys():
                masked_sequence_tokens[key] = masked_sequence_tokens[key].squeeze(0)
            for key in groundtruth_sequence_tokens.keys():
                groundtruth_sequence_tokens[key] = groundtruth_sequence_tokens[key].squeeze(0)
            return masked_sequence_tokens, groundtruth_sequence_tokens
        else: # We need to mask the sentence
            sequence = self.data.iloc[index][0]
            groundtruth_sequence_tokens = self.tokenizer(sequence, padding='max_length', max_length=self.sequence_length,
                                                    truncation=True, return_tensors="pt")
            for key in groundtruth_sequence_tokens.keys(): 
                # delete the first dim of tensor (i.e. (1, N) -> (N)
                groundtruth_sequence_tokens[key] = groundtruth_sequence_tokens[key].squeeze(0)
            # clone input ids tensor (dont want to overwrite our ground truth input ids)
            masked_sequence_tokens = copy.deepcopy(groundtruth_sequence_tokens) # copy dict and replace input ids
            
            masked_sequence_input_ids = self.__random_masking__(masked_sequence_tokens['input_ids'], self.mask_prob)
            masked_sequence_tokens['input_ids'] = masked_sequence_input_ids
            return masked_sequence_tokens, groundtruth_sequence_tokens
    def __len__(self):
        """
        Returns length of dataset
        """
        return len(self.data)