import os
from os.path import join, exists

import matplotlib.pyplot as plt

from monai.data import decollate_batch
from monai.transforms import EnsureType, AsDiscrete, KeepLargestConnectedComponent, Compose, SaveImage


def saveimage(batch_data, outputs, device, cf, args, epoch, test=False):

    if test:
        if hasattr(args, 'nfold'):
            if not exists(join(args.pdir, cf.export_params.save_dir, args.job, 'best_outputs/ensemble',
                               f'{epoch + 1:03d}')):
                os.makedirs(join(args.pdir, cf.export_params.save_dir, args.job, 'best_outputs/ensemble',
                                 f'{epoch + 1:03d}'))
            odir = join(args.pdir, cf.export_params.save_dir, args.job, 'best_outputs/ensemble',
                        f'{epoch + 1:03d}')
        else:
            if not exists(join(args.pdir, cf.export_params.save_dir, args.job, 'best_outputs', f'f{args.fold}',
                               f'{epoch + 1:03d}')):
                os.makedirs(join(args.pdir, cf.export_params.save_dir, args.job, 'best_outputs', f'f{args.fold}',
                                 f'{epoch + 1:03d}'))
            odir = join(args.pdir, cf.export_params.save_dir, args.job, 'best_outputs', f'f{args.fold}',
                        f'{epoch + 1:03d}')

    else:
        if not exists(join(args.pdir, cf.export_params.save_dir, args.job, f'f{args.fold}', f'{epoch + 1:03d}')):
            os.makedirs(join(args.pdir, cf.export_params.save_dir, args.job, f'f{args.fold}', f'{epoch + 1:03d}'))
        odir = join(args.pdir, cf.export_params.save_dir, args.job, f'f{args.fold}', f'{epoch + 1:03d}')

    inputs = batch_data["image"].to(device)
    labels = batch_data["label"].to(device)
    inputs_md = batch_data["image_meta_dict"]
    labels_md = batch_data["label_meta_dict"]

    inputs = [i for i in decollate_batch(inputs)]
    inputs_md = [i for i in decollate_batch(inputs_md)]
    labels = [i for i in decollate_batch(labels)]
    labels_md = [i for i in decollate_batch(labels_md)]

    if test:
        l = [i for i in range(1, cf.model_params.out_channels)]
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True),
                             KeepLargestConnectedComponent(applied_labels=l)])

    else:
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True)])

    outputs = [post_pred(i) for i in decollate_batch(outputs)]

    for input, input_md, label, label_md, output in zip(inputs, inputs_md, labels, labels_md, outputs):

        SaveImage(output_dir=odir,
                  output_postfix='img', output_ext='.nii.gz', resample=True, mode='bilinear', scale=255,
                  separate_folder=False, print_log=False)(input, meta_data=input_md)

        SaveImage(output_dir=odir,
                  output_postfix='gt', output_ext='.nii.gz', resample=True, mode='nearest', scale=255,
                  separate_folder=False, print_log=False)(label, meta_data=label_md)

        SaveImage(output_dir=odir,
                  output_postfix='pred', output_ext='.nii.gz', resample=True, mode='nearest', scale=255,
                  separate_folder=False, print_log=False)(output, meta_data=label_md)


def plotloss_metrics(cf, args, epoch_loss_values, val_loss_values, metric_values):

    if not exists(join(args.pdir, cf.export_params.save_dir, args.job, f'f{args.fold}')):
        os.makedirs(join(args.pdir, cf.export_params.save_dir, args.job, f'f{args.fold}'))

    plt.subplot(1, 2, 1)
    plt.title("Loss")
    x0 = [i + 1 for i in range(len(epoch_loss_values))]
    x1 = [i + 1 for i in range(1, len(epoch_loss_values), 10)]
    y = epoch_loss_values
    y_val = val_loss_values
    plt.xlabel("epoch")
    plt.grid(alpha=0.4, linestyle=":")
    plt.plot(x0, y, color="green", label="epoch loss")
    plt.plot(x1, y_val, color="blue", label="val loss")
    plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.grid(alpha=0.4, linestyle=":")
    for i in range(len(metric_values[0])):
        y = [metrics[i] for metrics in metric_values]
        plt.plot(x1, y, label=f'label{i}')
    plt.legend(loc='upper right')
    plt.savefig(join(args.pdir, cf.export_params.save_dir, args.job, f'f{args.fold}', 'training.png'), format='png')
