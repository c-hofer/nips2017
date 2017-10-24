import torch
import torch.nn as nn
import numpy as np

from torch import optim

from ..sharedCode.provider import Provider
from ..sharedCode.experiments import train_test_from_dataset, \
    UpperDiagonalThresholdedLogTransform, \
    pers_dgm_center_init,\
    SLayerPHT


import chofer_torchex.utils.trainer as tr
from chofer_torchex.utils.trainer.plugins import *


def _parameters():
    return \
    {
        'data_path': None,
        'epochs': 300,
        'momentum': 0.7,
        'lr_start': 0.1,
        'lr_ep_step': 20,
        'lr_adaption': 0.5,
        'test_ratio': 0.1,
        'batch_size': 128,
        'cuda': False
    }


def _data_setup(params):
    view_name_template = 'dim_0_dir_{}'
    subscripted_views = sorted([view_name_template.format(i) for i in range(0, 32)])
    assert (str(len(subscripted_views)) in params['data_path'])

    print('Loading provider...')
    dataset = Provider()
    dataset.read_from_h5(params['data_path'])

    assert all(view_name in dataset.view_names for view_name in subscripted_views)

    print('Create data loader...')
    data_train, data_test = train_test_from_dataset(dataset,
                                                    test_size=params['test_ratio'],
                                                    batch_size=params['batch_size'])

    return data_train, data_test, subscripted_views


class MyModel(torch.nn.Module):
    def __init__(self, subscripted_views):
        super(MyModel, self).__init__()
        self.subscripted_views = subscripted_views

        n_elements = 75
        n_filters = 16
        stage_2_out = 25
        n_neighbor_directions = 1

        self.transform = UpperDiagonalThresholdedLogTransform(0.1)

        self.pht_sl = SLayerPHT(len(subscripted_views),
                                n_elements,
                                2,
                                n_neighbor_directions=n_neighbor_directions,
                                center_init=self.transform(pers_dgm_center_init(n_elements)),
                                sharpness_init=torch.ones(n_elements, 2) * 4)

        self.stage_1 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('conv_1', nn.Conv1d(1 + 2 * n_neighbor_directions, n_filters, 1, bias=False))
            seq.add_module('conv_2', nn.Conv1d(n_filters, 4, 1, bias=False))
            self.stage_1.append(seq)
            self.add_module('stage_1_{}'.format(i), seq)

        self.stage_2 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('linear_1', nn.Linear(n_elements, stage_2_out))
            seq.add_module('batch_norm', nn.BatchNorm1d(stage_2_out))
            seq.add_module('linear_2'
                           , nn.Linear(stage_2_out, stage_2_out))
            seq.add_module('relu', nn.ReLU())
            seq.add_module('Dropout', nn.Dropout(0.4))

            self.stage_2.append(seq)
            self.add_module('stage_2_{}'.format(i), seq)

        linear_1 = nn.Sequential()
        linear_1.add_module('linear', nn.Linear(len(subscripted_views) * stage_2_out, 100))
        linear_1.add_module('batchnorm', torch.nn.BatchNorm1d(100))
        linear_1.add_module('drop_out', torch.nn.Dropout(0.3))
        self.linear_1 = linear_1

        linear_2 = nn.Sequential()
        linear_2.add_module('linear', nn.Linear(100, 70))

        self.linear_2 = linear_2

    def forward(self, batch):
        x = [batch[n] for n in self.subscripted_views]
        x = [[self.transform(dgm) for dgm in view_batch] for view_batch in x]

        x = self.pht_sl(x)

        x = [l(xx) for l, xx in zip(self.stage_1, x)]

        x = [torch.squeeze(torch.max(xx, 1)[0]) for xx in x]

        x = [l(xx) for l, xx in zip(self.stage_2, x)]

        x = torch.cat(x, 1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


def _create_trainer(model, params, data_train, data_test):
    optimizer = optim.SGD(model.parameters(),
                          lr=params['lr_start'],
                          momentum=params['momentum'])

    loss = nn.CrossEntropyLoss()

    trainer = tr.Trainer(model=model,
                         optimizer=optimizer,
                         loss=loss,
                         train_data=data_train,
                         n_epochs=params['epochs'],
                         cuda=params['cuda'],
                         variable_created_by_model=True)

    def determine_lr(self, **kwargs):
        epoch = kwargs['epoch_count']
        if epoch % params['lr_ep_step'] == 0:
            return params['lr_start'] / 2 ** (epoch / params['lr_ep_step'])

    lr_scheduler = LearningRateScheduler(determine_lr, verbose=True)
    lr_scheduler.register(trainer)

    progress = ConsoleBatchProgress()
    progress.register(trainer)

    prediction_monitor_test = PredictionMonitor(data_test,
                                                verbose=True,
                                                eval_every_n_epochs=1,
                                                variable_created_by_model=True)
    prediction_monitor_test.register(trainer)
    trainer.prediction_monitor = prediction_monitor_test

    return trainer


def experiment(data_path):
    params = _parameters()
    params['data_path'] = data_path

    if torch.cuda.is_available():
        params['cuda'] = True

    print('Data setup...')
    data_train, data_test, subscripted_views = _data_setup(params)

    print('Create model...')
    model = MyModel(subscripted_views)

    print('Setup trainer...')
    trainer = _create_trainer(model, params, data_train, data_test)
    print('Starting...')
    trainer.run()

    last_10_accuracies = list(trainer.prediction_monitor.accuracies.values())[-10:]
    mean = np.mean(last_10_accuracies)

    return mean
