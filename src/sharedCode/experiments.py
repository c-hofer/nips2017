import torch
import numpy as np
import time
import shutil
import json
import numpy
import datetime
import os

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing.label import LabelEncoder
from collections import defaultdict
from torch.nn import Module
from chofer_torchex.nn import SLayer


class PersistenceDiagramProviderCollate:
    def __init__(self, provider, wanted_views: [str] = None,
                 label_map: callable = lambda x: x,
                 output_type=torch.FloatTensor,
                 target_type=torch.LongTensor):
        provided_views = provider.view_names

        if wanted_views is None:
            self.wanted_views = provided_views

        else:
            for wv in wanted_views:
                if wv not in provided_views:
                    raise ValueError('{} is not provided by {} which provides {}'.format(wv, provider, provided_views))

            self.wanted_views = wanted_views

        if not callable(label_map):
            raise ValueError('label_map is expected to be callable.')

        self.label_map = label_map

        self.output_type = output_type
        self.target_type = target_type

    def __call__(self, sample_target_iter):
        batch_views_unprepared, batch_views_prepared, targets = defaultdict(list), {}, []

        for dgm_dict, label in sample_target_iter:
            for view_name in self.wanted_views:
                dgm = list(dgm_dict[view_name])
                dgm = self.output_type(dgm)

                batch_views_unprepared[view_name].append(dgm)

            targets.append(self.label_map(label))

        targets = self.target_type(targets)

        return batch_views_unprepared, targets


class SubsetRandomSampler:
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def train_test_from_dataset(dataset,
                            test_size=0.2,
                            batch_size=64,
                            wanted_views=None):

    sample_labels = list(dataset.sample_labels)
    label_encoder = LabelEncoder().fit(sample_labels)
    sample_labels = label_encoder.transform(sample_labels)

    label_map = lambda l: int(label_encoder.transform([l])[0])
    collate_fn = PersistenceDiagramProviderCollate(dataset, label_map=label_map, wanted_views=wanted_views)

    sp = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    train_i, test_i = list(sp.split([0]*len(sample_labels), sample_labels))[0]

    data_train = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=False,
                            sampler=SubsetRandomSampler(train_i.tolist()))

    data_test = DataLoader(dataset,
                           batch_size=batch_size,
                           collate_fn=collate_fn,
                           shuffle=False,
                           sampler=SubsetRandomSampler(test_i.tolist()))

    return data_train, data_test


class UpperDiagonalThresholdedLogTransform:
    def __init__(self, nu):
        self.b_1 = (torch.Tensor([1, 1]) / np.sqrt(2))
        self.b_2 = (torch.Tensor([-1, 1]) / np.sqrt(2))
        self.nu = nu

    def __call__(self, dgm):
        if dgm.ndimension() == 0:
            return dgm

        if dgm.is_cuda:
            self.b_1 = self.b_1.cuda()
            self.b_2 = self.b_2.cuda()

        x = torch.mul(dgm, self.b_1.repeat(dgm.size(0), 1))
        x = torch.sum(x, 1).squeeze()
        y = torch.mul(dgm, self.b_2.repeat( dgm.size(0), 1))
        y = torch.sum(y, 1).squeeze()
        i = (y <= self.nu)
        y[i] = torch.log(y[i] / self.nu) + self.nu
        ret = torch.stack([x, y], 1)
        return ret


def pers_dgm_center_init(n_elements):
    centers = []
    while len(centers) < n_elements:
        x = np.random.rand(2)
        if x[1] > x[0]:
            centers.append(x.tolist())

    return torch.Tensor(centers)


class SLayerPHT(Module):
    def __init__(self,
                 n_directions,
                 n_elements,
                 point_dim,
                 n_neighbor_directions=0,
                 center_init=None,
                 sharpness_init=None):
        super(SLayerPHT, self).__init__()

        self.n_directions = n_directions
        self.n_elements = n_elements
        self.point_dim = point_dim
        self.n_neighbor_directions = n_neighbor_directions

        self.slayers = [SLayer(n_elements, point_dim, center_init, sharpness_init)
                        for i in range(n_directions)]
        for i, l in enumerate(self.slayers):
            self.add_module('sl_{}'.format(i), l)

    def forward(self, input):
        assert len(input) == self.n_directions

        prepared_batches = None
        if all(SLayer.is_prepared_batch(b) for b in input):
            prepared_batches = input
        elif all(SLayer.is_list_of_tensors(b) for b in input):
            prepared_batches = [SLayer.prepare_batch(input_i, self.point_dim) for input_i in input]
        else:
            raise ValueError('Unrecognized input format! Expected list of Tensors or list of SLayer.prepare_batch outputs!')

        batch_size = prepared_batches[0][0].size()[0]
        assert all(prep_b[0].size()[0] == batch_size for prep_b in prepared_batches)

        output = []
        for i, sl in enumerate(self.slayers):
            i_th_output = []
            i_th_output.append(sl(prepared_batches[i]))

            for j in range(1, self.n_neighbor_directions + 1):
                i_th_output.append(sl(prepared_batches[i - j]))
                i_th_output.append(sl(prepared_batches[(i + j) % self.n_directions]))

            if self.n_directions > 0:
                i_th_output = torch.stack(i_th_output, 1)
            else:
                i_th_output = output[0]

            output.append(i_th_output)

        return output

    @property
    def is_gpu(self):
        return self.slayers[0].is_gpu


def reduce_essential_dgm(dgm):

    if dgm.ndimension() == 0:
        return dgm
    else:
        return dgm[:, 0].contiguous().view(-1, 1)
