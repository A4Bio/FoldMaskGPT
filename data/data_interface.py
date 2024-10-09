import inspect

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .collate_fn import tensor_dict_stack_padding_collater
from .datasets import PDBVQDataset


class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        print("batch_size", self.batch_size)
        self.load_data_module()
        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage=None, mask_location=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.trainset is None:
                self.trainset = self.instancialize(split='train')

            if self.valset is None:
                self.valset = self.instancialize(split='test')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            if self.testset is None:
                self.testset = self.instancialize(split='test')

        if stage == 'scaffold' or stage is None:
            if self.testset is None:
                self.testset = self.instancialize(split='scaffold')

    def train_dataloader(self, db=None, preprocess=False):
        collate_fn = tensor_dict_stack_padding_collater(
            padding_id={
                "input_ids": self.trainset.tokenizer.vocab['[PAD]'],  # 🔍
                "label_mask": 0,
                "pos_ids": 0,
                "labels": -100,
            },
            padding_position="right",
            return_padding_mask=True,
            tensor_keys_to_create_mask=["input_ids"],
            exclude_none=True,
        )
        return self.instancialize_module(DataLoader, dataset=self.trainset, shuffle=True, prefetch_factor=4, pin_memory=True, collate_fn=collate_fn)

    def val_dataloader(self, db=None, preprocess=False):
        collate_fn = tensor_dict_stack_padding_collater(
            padding_id={
                "input_ids": self.valset.tokenizer.vocab['[PAD]'],  # 🔍
                "label_mask": 0,
                "pos_ids": 0,
                "labels": -100,
            },
            padding_position="right",
            return_padding_mask=True,
            tensor_keys_to_create_mask=["input_ids"],
            exclude_none=True,
        )
        return self.instancialize_module(DataLoader, dataset=self.valset, batch_size=self.hparams.valid_batch_size, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self, db=None, preprocess=False, mask_location=None):
        collate_fn = tensor_dict_stack_padding_collater(
            padding_id={
                "input_ids": self.testset.tokenizer.vocab['[PAD]'],  # 🔍
                "label_mask": 0,
                "pos_ids": 0,
                "labels": -100,
            },
            padding_position="right",
            return_padding_mask=True,
            tensor_keys_to_create_mask=["input_ids"],
            exclude_none=True,
        )
        return self.instancialize_module(DataLoader, dataset=self.testset, batch_size=self.hparams.valid_batch_size, shuffle=False, collate_fn=collate_fn)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        # 🔍 add attention_mask to inputs
        if batch is None:
            return None
        else:
            inputs, masks = batch
            inputs["attention_mask"] = masks["input_ids"]
            return inputs

    def load_data_module(self):
        self.data_module = PDBVQDataset

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = list(inspect.signature(self.data_module.__init__).parameters)[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)

    def instancialize_module(self, module, **other_args):
        class_args = list(inspect.signature(module.__init__).parameters)[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return module(**args1)
