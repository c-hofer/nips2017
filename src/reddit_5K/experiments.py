import torch
import torch.nn as nn
import numpy as np

from torch import optim

from ..sharedCode.provider import Provider
from ..sharedCode.experiments import train_test_from_dataset, \
    UpperDiagonalThresholdedLogTransform, \
    pers_dgm_center_init, \
    reduce_essential_dgm

from chofer_torchex.nn import SLayer
import chofer_torchex.utils.trainer as tr
from chofer_torchex.utils.trainer.plugins import *


def _parameters():
    return \
    {
        'data_path': None,
        'epochs': 300,
        'momentum': 0.5,
        'lr_start': 0.1,
        'lr_ep_step': 20,
        'lr_adaption': 0.5,
        'test_ratio': 0.1,
        'batch_size': 128,
        'cuda': False
    }


def _data_setup(params):
    subscripted_views = ['DegreeVertexFiltration_dim_0',
                         'DegreeVertexFiltration_dim_0_essential',
                         'DegreeVertexFiltration_dim_1_essential'
                         ]

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
        self.transform = UpperDiagonalThresholdedLogTransform(0.1)

        def get_init(n_elements):
            transform = UpperDiagonalThresholdedLogTransform(0.1)
            return transform(pers_dgm_center_init(n_elements))

        self.dim_0 = SLayer(150, 2, get_init(150), torch.ones(150, 2) * 3)
        self.dim_0_ess = SLayer(50, 1)
        self.dim_1_ess = SLayer(50, 1)
        self.slayers = [self.dim_0,
                        self.dim_0_ess,
                        self.dim_1_ess
                        ]

        self.stage_1 = []
        stage_1_outs = [75, 25, 25]

        for i, (n_in, n_out) in enumerate(zip([150, 50, 50], stage_1_outs)):
            seq = nn.Sequential()
            seq.add_module('linear_1', nn.Linear(n_in, n_out))
            seq.add_module('batch_norm', nn.BatchNorm1d(n_out))
            seq.add_module('drop_out_1', nn.Dropout(0.1))
            seq.add_module('linear_2', nn.Linear(n_out, n_out))
            seq.add_module('relu', nn.ReLU())
            seq.add_module('drop_out_2', nn.Dropout(0.1))

            self.stage_1.append(seq)
            self.add_module('stage_1_{}'.format(i), seq)

        linear_1 = nn.Sequential()
        linear_1.add_module('linear_1', nn.Linear(sum(stage_1_outs), 200))
        linear_1.add_module('batchnorm_1', torch.nn.BatchNorm1d(200))
        linear_1.add_module('relu_1', nn.ReLU())
        linear_1.add_module('linear_2', nn.Linear(200, 100))
        linear_1.add_module('batchnorm_2', torch.nn.BatchNorm1d(100))
        linear_1.add_module('drop_out_2', torch.nn.Dropout(0.1))
        linear_1.add_module('relu_2', nn.ReLU())
        linear_1.add_module('linear_3', nn.Linear(100, 50))
        linear_1.add_module('batchnorm_3', nn.BatchNorm1d(50))
        linear_1.add_module('relu_3', nn.ReLU())
        linear_1.add_module('linear_4', nn.Linear(50, 5))
        linear_1.add_module('batchnorm_4', nn.BatchNorm1d(5))
        self.linear_1 = linear_1

    def forward(self, batch):
        x = [batch[n] for n in self.subscripted_views]

        x = [
             [self.transform(dgm) for dgm in x[0]],
             [reduce_essential_dgm(dgm) for dgm in x[1]],
             [reduce_essential_dgm(dgm) for dgm in x[2]]
            ]

        x_sl = [l(xx) for l, xx in zip(self.slayers, x)]

        x = [l(xx) for l, xx in zip(self.stage_1, x_sl)]

        x = torch.cat(x, 1)

        x = self.linear_1(x)

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
