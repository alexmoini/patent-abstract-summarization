from summarization_utils import SummarizationDataModule
from summarization_tuner import SummarizationModel
import unittest
import transformers
import torch
from summary_loader import load_and_split
import argparse
import os
import shutil


class TestSummarizationDataModule(unittest.TestCase):
    
    def setUp(self):
        """
        Set up the test case
        """
        args = argparse.Namespace()
        args.split = 'validation'
        args.output_dir = 'summarization_test_data/'
        args.dataset_name = 'big_patent'
        args.dataset_config_name = 'd'
        args.test_data = True

        load_and_split(args)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.max_length = 1024
        self.dataset_path = 'summarization_test_data/unittest_data'
        self.data = SummarizationDataModule(self.dataset_path, self.tokenizer, \
                                            self.max_length)
        
    def test_getitem(self):
        """
        Test that the getitem function returns a dictionary with the correct keys
        """
        item = self.data[0]
        self.assertEqual(item.keys(), {'input_ids', 'attention_mask', 'labels'})
        self.assertIsInstance(item['input_ids'], torch.Tensor)
        self.assertIsInstance(item['attention_mask'], torch.Tensor)
        self.assertIsInstance(item['labels'], torch.Tensor)
        
    def test_len(self):
        """
        Test that the length of the dataset is correct
        """
        self.assertEqual(len(self.data), 5)
        
    def tearDown(self):
        """
        Clean up after the test case
        """
        # open summarization_test_data and delete all files but keep the directory
        for file in os.listdir('summarization_test_data'):
                os.remove(os.path.join('summarization_test_data', file))

class TestSummarizationModel(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case
        """
        args = argparse.Namespace()
        args.split = 'validation'
        args.output_dir = 'summarization_test_data/'
        args.dataset_name = 'big_patent'
        args.dataset_config_name = 'd'
        args.test_data = True

        load_and_split(args)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.max_length = 1024
        self.dataset_path = 'summarization_test_data/unittest_data'

        self.data = SummarizationDataModule(self.dataset_path, self.tokenizer, \
                                            self.max_length, '')


        self.model = SummarizationModel('facebook/bart-large-cnn')
        
    def test_forward(self):
        """
        Test that the forward function returns a dictionary with the correct keys
        """
        item = self.data[0]
        output = self.model(item['input_ids'], item['attention_mask'], labels=item['labels'])
        self.assertIsInstance(output['loss'], torch.Tensor)
        self.assertIsInstance(output['logits'], torch.Tensor)
        
    def test_training_step(self):
        """
        Test that the training step function returns a dictionary 
        with the correct keys
        """
        item = self.data[0]
        output = self.model.training_step(item, 0)
        # output should be torch tensor with gradient
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(output.requires_grad)
        
    def test_validation_step(self):
        """
        Test that the validation step function returns a dictionary 
        with the correct keys
        """
        item = self.data[0]
        output = self.model.validation_step(item, 0)
        # output should be torch tensor without gradient
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(output.requires_grad)
        
    def test_configure_optimizers(self):
        """
        Test that the configure optimizers function returns a dictionary 
        with the correct keys
        """
        output = self.model.configure_optimizers()
        self.assertEqual(output.keys(), {'optimizer', 'lr_scheduler'})
        self.assertIsInstance(output['optimizer'], torch.optim.Optimizer)
        self.assertIsInstance(output['lr_scheduler']['monitor'], str)
        
    def test_predict(self):
        """
        Test that the predict function returns a dictionary with 
        the correct keys
        """
        tokens = self.data[0]['input_ids'].tolist()
        text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        output = self.model.predict(text[0])
        self.assertIsInstance(output, list)
        
    def test_save_load_model(self):
        """
        Test that the save and load model functions work
        """
        self.model.save_model('summarization_test_data/test_model')
        self.bart = SummarizationModel('facebook/bart-large-cnn', 

                                       checkpoint_dir=\
                                       'summarization_test_data/test_model')
    def tearDown(self):
        """
        Clean up after the test case
        """
        # open summarization_test_data and delete all files
        # but keep the directory
        for file in os.listdir('summarization_test_data'):
                os.remove(os.path.join('summarization_test_data', file))


if __name__ == '__main__':
    unittest.main()
