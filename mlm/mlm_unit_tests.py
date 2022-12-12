from embedding_backbone import TextEmbeddingBackbone
from mlm_tuner import MaskedLanguageModelingModel
from mlm_utils import CorpusMaskingDataset
import os
import unittest
import transformers
import torch
import argparse
from mlm_corpus_loader import load_and_split

class TestMaskedLanguageModelingDataModule(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case
        """

        # load in dummy data
        args = argparse.Namespace()
        args.split = 'validation'
        args.output_dir = 'mlm_test_data/'
        args.dataset_name = 'big_patent'
        args.dataset_config_name = 'd'
        args.test_data = True

        load_and_split(args)


        # initialize the data module
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 128
        self.dataset_path = 'mlm_test_data/unittest_data'
        self.data = CorpusMaskingDataset(self.dataset_path, self.tokenizer)
    def test_getitem(self):
        """
        Test that the getitem function returns a dictionary with the correct keys
        """
        masked, ground_truth = self.data[0]

        # type checking
        self.assertEqual(masked.keys(), {'input_ids', 'attention_mask', 'token_type_ids'})
        self.assertEqual(ground_truth.keys(), {'input_ids', 'attention_mask', 'token_type_ids'})
        self.assertIsInstance(masked['input_ids'], torch.Tensor)
        self.assertIsInstance(masked['attention_mask'], torch.Tensor)
        self.assertIsInstance(masked['token_type_ids'], torch.Tensor)
        self.assertIsInstance(ground_truth['input_ids'], torch.Tensor)
        self.assertIsInstance(ground_truth['attention_mask'], torch.Tensor)
        self.assertIsInstance(ground_truth['token_type_ids'], torch.Tensor)

    def test_len(self):
        """
        Test that the length of the dataset is correct
        """
        self.assertEqual(len(self.data), 5)
    def tearDown(self):
        """
        Clean up after the test case
        """
        # open mlm_test_data and delete all files but keep the directory
        for file in os.listdir('mlm_test_data'):
                os.remove(os.path.join('mlm_test_data', file))

class TestMaskedLanguageModelingModel(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case
        """

        # load in dummy data
        args = argparse.Namespace()
        args.split = 'validation'
        args.output_dir = 'mlm_test_data/'
        args.dataset_name = 'big_patent'
        args.dataset_config_name = 'd'
        args.test_data = True

        load_and_split(args)


        # initialize the data module and model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 128
        self.dataset_path = 'mlm_test_data/unittest_data'
        self.data = CorpusMaskingDataset(self.dataset_path, self.tokenizer)
        self.backbone = TextEmbeddingBackbone('bert-base-uncased')
        self.model = MaskedLanguageModelingModel(self.backbone)
    def test_forward(self):
        """
        Test that the forward function returns a dictionary with the correct keys
        """
        masked, _ = self.data[0]
        input_ids = masked['input_ids']
        attention_mask = masked['attention_mask']
        output = self.model(input_ids, attention_mask)

        # type checking
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 512, self.tokenizer.vocab_size))

    def test_training_step(self):
        """
        Test that the training step function returns a dictionary with the correct keys
        """
        loss = self.model.training_step(self.data[0], 0)

        # type checking
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)

    def test_validation_step(self):
        """
        Test that the validation step function returns a dictionary with the correct keys
        """
        loss = self.model.validation_step(self.data[0], 0)

        # type checking
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(loss.requires_grad)

    def test_save_backbone(self):
        """
        Test that the save backbone function saves the backbone to the correct path
        """
        self.model.save_backbone('mlm_test_data/backbone.pt')
        self.assertTrue(os.path.exists('mlm_test_data/backbone.pt'))

    def tearDown(self):
        """
        Clean up after the test case
        """
        # open mlm_test_data and delete all files but keep the directory
        for file in os.listdir('mlm_test_data'):
                os.remove(os.path.join('mlm_test_data', file))

class TestTextEmbeddingBackbone(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case
        """
        args = argparse.Namespace()
        args.split = 'validation'
        args.output_dir = 'mlm_test_data/'
        args.dataset_name = 'big_patent'
        args.dataset_config_name = 'd'
        args.test_data = True

        load_and_split(args)
        
        self.model_architecture = 'bert-base-uncased'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_architecture)
        self.max_length = 128
        self.dataset_path = 'mlm_test_data/unittest_data'
        self.data = CorpusMaskingDataset(self.dataset_path, self.tokenizer)
        self.backbone = TextEmbeddingBackbone(self.model_architecture)
    def test_forward(self):
        """
        Test that the forward function returns a dictionary with the correct keys
        """
        item, _ = self.data[0]
        output = self.backbone(item['input_ids'], item['attention_mask'])
        
        # type checking
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 512, 768))

    def tearDown(self):
        """
        Clean up after the test case
        """
        # open mlm_test_data and delete all files but keep the directory
        for file in os.listdir('mlm_test_data'):
                os.remove(os.path.join('mlm_test_data', file))