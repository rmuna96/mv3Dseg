import argparse
import os
import yaml
import json
from os.path import join, exists

import random
import numpy as np
from sklearn.model_selection import KFold


def configge(args):
    """
        Create a json file with all the configurations for training.
        Split dataset according to splitting scheme adopted in NTNU.
    """

    with open(join(args.idir, 'ge', f'{args.ymlfile}.yml'), 'r') as ifile:
        split = yaml.safe_load(ifile)

    with open(join(args.idir, 'ge/dataset.json'), 'r') as ifile:
        dataset = json.load(ifile)

    train_id = []
    validation_id = []
    test_id = []

    for file in split['train']['files']:
        fname = file[0]

        for key in dataset.keys():
            if fname in dataset[key]['filename']:
                idx = dataset[key]['filename'].index(fname)
                train_id.append(int(dataset[key]['id'][idx]))

    for file in split['validation']['files']:
        fname = file[0]

        for key in dataset.keys():
            if fname in dataset[key]['filename']:
                idx = dataset[key]['filename'].index(fname)
                validation_id.append(int(dataset[key]['id'][idx]))

    for file in split['test']['files']:
        fname = file[0]

        for key in dataset.keys():
            if fname in dataset[key]['filename']:
                idx = dataset[key]['filename'].index(fname)
                test_id.append(int(dataset[key]['id'][idx]))

    cfg = {
            "import_params": {
                "image_dir": "inputs/ge/images",
                "mask_dir": "inputs/ge/masks",
            },
            "export_params": {
                "save_dir": "outputs/ge"
            },
            "training_params": {
                "learning_rate": args.lr,
                "device": "cuda",
                "batch_size": args.bs,
                "num_epochs": args.epochs,
                "num_workers": args.workers,
                "load_model": False,
                "save_interval": args.savepoch,
                "flag_schedule": args.scheduler,
            },
            "scheduler_params": {
                "schedulers": {
                    "StepLR": {
                        "step_size": args.stepsize,
                        "gamma": 0.1
                    },
                    "OneCycleLR": {
                        "max_lr": 0.01,
                        "steps_per_epoch": 90,
                        "tot_epoch": 1000
                    }
                },
                "scheduler": "StepLR"
            },
            "loss_params": {
                "lambda_dice": args.lambdadice,
                "lambda_focal": args.lambdafocal,
            },
            "transform_params": {
                "crop_size": [args.cropsize for i in range(3)]
            },
            "model_params": {
                "in_channels": 1,
                "out_channels": args.outchannel,
                "features": [
                    16,
                    32,
                    64,
                    128,
                    256
                ]
            },
            'dataset_params': {
                'index_train': sorted(train_id),
                'index_val': sorted(validation_id),
                'index_test': sorted(test_id),
            }

    }

    if not exists(join(args.odir, 'ge', args.job)):
        os.makedirs(join(args.odir, 'ge', args.job))

    with open(join(args.odir, 'ge', args.job, f'f{args.fold}.json'), 'w') as outfile:
        json.dump(cfg, outfile, indent=4)


def config_b(args):
    """
        Create a json file with all the configurations for training.
        Split dataset according to the splitting rule given as input.
    """
    assert (args.rtrain + args.rval + args.rtest) == 1, \
        "Train, validation and test ratios must sum to 1."

    with open(join(args.idir, 'philips/dataset.json'), 'r') as ifile:
        dataset = json.load(ifile)

    ptsids = list(dataset.keys())
    n = len(ptsids)
    ptids_idxs = [idx for idx in range(n)]
    trsamp, valsamp = int(args.rtrain * n), int((args.rtrain + args.rval) * n)
    random.shuffle(ptids_idxs)
    tridx, validx, testidx = ptids_idxs[:trsamp], ptids_idxs[trsamp:valsamp], ptids_idxs[valsamp:]

    cfg = {
        "import_params": {
            "image_dir": "inputs/philips/images",
            "mask_dir": "inputs/philips/masks",
        },
        "export_params": {
            "save_dir": "outputs/philips"
        },
        "training_params": {
            "learning_rate": args.lr,
            "device": "cuda",
            "batch_size": args.bs,
            "num_epochs": args.epochs,
            "num_workers": args.workers,
            "load_model": False,
            "save_interval": args.savepoch,
            "flag_schedule": args.scheduler
        },
        "scheduler_params": {
            "schedulers": {
                "StepLR": {
                    "step_size": args.stepsize,
                    "gamma": 0.1
                },
                "OneCycleLR": {
                    "max_lr": 0.01,
                    "steps_per_epoch": 90,
                    "tot_epoch": 1000
                }
            },
            "scheduler": "StepLR"
        },
        "loss_params": {
            "lambda_dice": args.lambdadice,
            "lambda_focal": args.lambdafocal,
        },
        "transform_params": {
            "crop_size": [args.cropsize for _ in range(3)]
        },
        "model_params": {
            "in_channels": 1,
            "out_channels": args.outchannel,
            "features": [
                16,
                32,
                64,
                128,
                256
            ]
        },
    }

    if args.splitds:

        cfg['dataset_params'] = {
            'index_train': sorted([int(id) for idx in tridx for id in dataset[ptsids[idx]]['id']]),
            'index_val': sorted([int(id) for idx in validx for id in dataset[ptsids[idx]]['id']]),
            'index_test': sorted([int(id) for idx in testidx for id in dataset[ptsids[idx]]['id']]),
        }

        if not exists(join(args.odir, 'philips', args.job)):
            os.makedirs(join(args.odir, 'philips', args.job))

        with open(join(args.odir, 'philips', args.job, f'f{args.fold}.json'), 'w') as outfile:
            json.dump(cfg, outfile, indent=4)

    else:

        with open(join(args.odir, 'philips', args.job, f'f{args.fold}.json'), 'r') as outfile:
            cfg_old = json.load(outfile)

        cfg_new = {cfg.get(k, k): v for k, v in cfg_old.items()}

        with open(join(args.odir, 'philips', args.job, f'f{args.fold}.json'), 'w') as outfile:
            json.dump(cfg_new, outfile, indent=4)


def config(args):
    """
        Create a json file with all the configuration for training.
        Split dataset according to splitting scheme given as input
        and the k-fold xvalidation specified.
    """

    assert (args.rtrain + args.rval + args.rtest) == 1, \
        "Train, validation and test ratios must sum to 1."

    with open(join(args.idir, 'philips/dataset.json'), 'r') as ifile:
        dataset = json.load(ifile)

    ptsids = list(dataset.keys())
    n = len(ptsids)
    ptids_idxs = [idx for idx in range(n)]

    cfg = {
        "import_params": {
            "image_dir": "inputs/philips/images",
            "mask_dir": "inputs/philips/masks",
        },
        "export_params": {
            "save_dir": "outputs/philips"
        },
        "training_params": {
            "learning_rate": args.lr,
            "device": "cuda",
            "batch_size": args.bs,
            "num_epochs": args.epochs,
            "num_workers": args.workers,
            "load_model": False,
            "save_interval": args.savepoch,
            "flag_schedule": args.scheduler
        },
        "scheduler_params": {
            "schedulers": {
                "StepLR": {
                    "step_size": args.stepsize,
                    "gamma": 0.1
                },
                "OneCycleLR": {
                    "max_lr": 0.01,
                    "steps_per_epoch": 90,
                    "tot_epoch": 1000
                }
            },
            "scheduler": "StepLR"
        },
        "loss_params": {
            "lambda_dice": args.lambdadice,
            "lambda_focal": args.lambdafocal,
        },
        "transform_params": {
            "crop_size": [args.cropsize for _ in range(3)]
        },
        "model_params": {
            "in_channels": 1,
            "out_channels": args.outchannel,
            "features": [
                16,
                32,
                64,
                128,
                256
            ]
        },
    }

    if args.job == 'xval':

        trvalsamp = int(n * (args.rtrain + args.rval))
        random.shuffle(ptids_idxs)
        trvalidx, testidx = ptids_idxs[0:trvalsamp], ptids_idxs[trvalsamp:]

        ### k-fold cross validation ###
        k = 5
        n_splits = int(len(trvalidx) / int(n * args.rval))

        kf = KFold(n_splits=n_splits, shuffle=True)

        folds_train = []
        folds_val = []

        for train, val in kf.split(trvalidx):
            trainidx = list(sorted(np.delete(trvalidx, val)))
            validx = list(sorted(np.delete(trvalidx, train)))

            folds_train.append(trainidx)
            folds_val.append(validx)

        foldidxs = list(sorted(np.random.choice(n_splits, k, replace=False)))

        if args.splitds:

            if not exists(join(args.odir, 'philips', args.job)):
                os.makedirs(join(args.odir, 'philips', args.job))

            for i, foldidx in enumerate(foldidxs):

                cfg['dataset_params'] = {
                    'index_train': sorted([int(id) for idx in folds_train[foldidx] for id in dataset[ptsids[idx]]['id']]),
                    'index_val': sorted([int(id) for idx in folds_val[foldidx] for id in dataset[ptsids[idx]]['id']]),
                    'index_test': sorted([int(id) for idx in testidx for id in dataset[ptsids[idx]]['id']]),
                }

                with open(join(args.odir, 'philips', args.job, f'f{i}.json'), 'w') as outfile:
                    json.dump(cfg, outfile, indent=4)

        else:

            for i, foldidx in enumerate(foldidxs):

                with open(join(args.odir, 'philips', args.job, f'f{i}.json'), 'r') as outfile:
                    cfg_old = json.load(outfile)

                cfg_new = {cfg.get(k, k): v for k, v in cfg_old.items()}

                with open(join(args.odir, 'philips', args.job, f'f{i}.json'), 'w') as outfile:
                    json.dump(cfg_new, outfile, indent=4)

    else:

        trsamp, valsamp = int(args.rtrain * n), int((args.rtrain + args.rval) * n)
        random.shuffle(ptids_idxs)
        tridx, validx, testidx = ptids_idxs[:trsamp], ptids_idxs[trsamp:valsamp], ptids_idxs[valsamp:]

        if args.splitds:

            cfg['dataset_params'] = {
                'index_train': sorted([int(id) for idx in tridx for id in dataset[ptsids[idx]]['id']]),
                'index_val': sorted([int(id) for idx in validx for id in dataset[ptsids[idx]]['id']]),
                'index_test': sorted([int(id) for idx in testidx for id in dataset[ptsids[idx]]['id']]),
            }

            if not exists(join(args.odir, 'philips', args.job)):
                os.makedirs(join(args.odir, 'philips', args.job))

            with open(join(args.odir, 'philips', args.job, f'f{args.fold}.json'), 'w') as outfile:
                json.dump(cfg, outfile, indent=4)

        else:

            with open(join(args.odir, 'philips', args.job, f'f{args.fold}.json'), 'r') as outfile:
                cfg_old = json.load(outfile)

            cfg_new = {k: cfg.get(k, v) for k, v in cfg_old.items()}

            with open(join(args.odir, 'philips', args.job, f'f{args.fold}.json'), 'w') as outfile:
                json.dump(cfg_new, outfile, indent=4)


if __name__ == '__main__':

    def str2bool(v):
        """
            Workaround to pass boolean to argparse
        """
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Create config file and perform dataset splitting')
    parser.add_argument('idir', type=str, default='./', help='input directory for data')
    parser.add_argument('ymlfile', type=str, help='dataset input ge file')
    parser.add_argument('odir', type=str, default='./config', help='output directory')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--bs', type=int, default=3, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='num epochs')
    parser.add_argument('--workers', type=int, default=6, help='workers')
    parser.add_argument('--savepoch', type=int, default=100, help='how often save solution')
    parser.add_argument('--scheduler', type=bool, default=True, help='use scheduler')
    parser.add_argument('--stepsize', type=int, default=100, help='step size for StepLR')
    parser.add_argument('--lambdadice', type=float, default=1, help='lambda dice')
    parser.add_argument('--lambdafocal', type=float, default=0.4, help='lambda focal')
    parser.add_argument('--cropsize', type=int, default=160, help='crop size for input images')
    parser.add_argument('--outchannel', type=int, default=3, help='outchannels')
    parser.add_argument('--splitds', type=str2bool, default=False, help='update dataset splitting')
    parser.add_argument('--rtrain', type=int, default=0.7, help='train ratio')
    parser.add_argument('--rval', type=int, default=0.1, help='val ration')
    parser.add_argument('--rtest', type=int, default=0.2, help='test ratio')
    parser.add_argument('--job', type=str, default='attempts', help='[attempts, xval]')
    parser.add_argument('--fold', type=int, default=0, help='specify the fold to process')
    args = parser.parse_args()

    configge(args)
    config(args)
