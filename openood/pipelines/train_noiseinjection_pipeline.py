import numpy as np
import torch

import openood.utils.comm as comm
#from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger

class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_std=0.05):
        """
        Args:
            dataset: The original dataset (e.g., ImglistDataset).
            noise_std: Standard deviation of the Gaussian noise to be added.
        """
        self.dataset = dataset
        self.noise_std = noise_std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Fetch the sample from the underlying dataset

        sample = self.dataset.getitem(idx)
        if np.random.rand(1)<0.5: # original data as well.
            return sample
        # Generate multiplicative noise (mean of 1 and std around it)
        noise = 1.0 + torch.randn_like(sample['data']) * self.noise_std

        # Apply multiplicative noise to the 'data' field (image data)
        noisy_data = sample['data'] * noise
        sample['data'] = noisy_data

        if 'data_aux' in sample:
            noise_aux = 1.0 + torch.randn_like(sample['data_aux']) * self.noise_std
            noisy_aux_data = sample['data_aux'] * noise_aux
            sample['data_aux'] = noisy_aux_data
        return sample


from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.preprocessors.utils import get_preprocessor
from openood.utils.config import Config
import os
from numpy import load
from torch.utils.data import DataLoader
from openood.datasets.imglist_augmix_dataset import ImglistAugMixDataset
from openood.datasets.imglist_extradata_dataset import ImglistExtraDataDataset, TwoSourceSampler
from openood.datasets.imglist_dataset import ImglistDataset

def get_NI_dataloader(config: Config,noise_std=0.01):
    # prepare a dataloader dictionary
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config, split)
        # weak augmentation for data_aux
        data_aux_preprocessor = TestStandardPreProcessor(config)

        if split_config.dataset_class == 'ImglistExtraDataDataset':
            dataset = ImglistExtraDataDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor,
                extra_data_pth=split_config.extra_data_pth,
                extra_label_pth=split_config.extra_label_pth,
                extra_percent=split_config.extra_percent)

            batch_sampler = TwoSourceSampler(dataset.orig_ids,
                                             dataset.extra_ids,
                                             split_config.batch_size,
                                             split_config.orig_ratio)
            if split == 'train':
                dataset = NoisyDataset(dataset, noise_std=noise_std)
            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=dataset_config.num_workers,
            )
        elif split_config.dataset_class == 'ImglistAugMixDataset':
            dataset = ImglistAugMixDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            sampler = None
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
                split_config.shuffle = False
            if split == 'train':
                dataset = NoisyDataset(dataset, noise_std=noise_std)
            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler)
        else:
            CustomDataset = eval(split_config.dataset_class)
            dataset = CustomDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            sampler = None
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
                split_config.shuffle = False
            if split == 'train':
                dataset = NoisyDataset(dataset, noise_std=noise_std)
            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler)

        dataloader_dict[split] = dataloader
    return dataloader_dict

class TrainNoisedInputPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # get dataloader
        loader_dict = get_NI_dataloader(self.config,noise_std=self.config.pipeline.noise_std)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)
        if self.config.num_gpus * self.config.num_machines > 1:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, val_loader, self.config)
        evaluator = get_evaluator(self.config)

        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)
            print('Start training...', flush=True)

        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            if isinstance(train_loader.sampler,
                          torch.utils.data.distributed.DistributedSampler):
                train_loader.sampler.set_epoch(epoch_idx - 1)

            # train and eval the model
            if self.config.trainer.name == 'mos':
                net, train_metrics, num_groups, group_slices = \
                    trainer.train_epoch(epoch_idx)
                val_metrics = evaluator.eval_acc(net,
                                                 val_loader,
                                                 train_loader,
                                                 epoch_idx,
                                                 num_groups=num_groups,
                                                 group_slices=group_slices)
            elif self.config.trainer.name in [
                    'cider', 'npos', 'palm', 'reweightood'
            ]:
                net, train_metrics = trainer.train_epoch(epoch_idx)
                # cider, npos, and palm only trains the backbone
                # cannot evaluate ID acc without training the fc layer
                val_metrics = train_metrics
            else:
                net, train_metrics = trainer.train_epoch(epoch_idx)
                val_metrics = evaluator.eval_acc(net, val_loader, None,
                                                 epoch_idx)
            comm.synchronize()
            if comm.is_main_process():
                # save model and report the result
                recorder.save_model(net, val_metrics)
                recorder.report(train_metrics, val_metrics)

        if comm.is_main_process():
            recorder.summary()
            print(u'\u2500' * 70, flush=True)

            # evaluate on test set
            print('Start testing...', flush=True)

        test_metrics = evaluator.eval_acc(net, test_loader)

        if comm.is_main_process():
            print('\nComplete Evaluation, Last accuracy {:.2f}'.format(
                100.0 * test_metrics['acc']),
                  flush=True)
            print('Completed!', flush=True)
