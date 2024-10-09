import datetime
import os
import warnings

import math
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from data.data_interface import DInterface
from model.model_interface_maskgit import MInterfaceMaskGIT

seed_everything(0)
warnings.filterwarnings("ignore")




def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FoldMaskGPT')
    parser.add_argument('--data_root', default='./data/', type=str, metavar='N', help='data root directory')
    parser.add_argument('--min_length', default=40, type=int)  # 🔍
    parser.add_argument('--max_length', default=512, type=int)  # 🔍
    parser.add_argument('--epoch', default=3000, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='batch size')
    parser.add_argument('--valid_batch_size', default=512, type=int, metavar='N', help='batch size for validation')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N', help='learning rate')
    parser.add_argument('--lr_scheduler', default="cosine", choices=['onecycle', 'cosine', 'step'], type=str, help='learning rate scheduler')
    parser.add_argument('--num_workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--lr_decay_steps', default=1000, type=int)

    parser.add_argument('--mask_method', default='cosine', type=str)  # 🔍
    parser.add_argument('--ckpt_path', default='/storage/huyuqi/gzy/FoldMaskGPT/model_zoom/params.ckpt', type=str)  # 🔍

    parser.add_argument('--offline', default=1, type=int)
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    args = parser.parse_args()
    return args


def load_callbacks(args):
    callbacks = []
    logdir = str(os.path.join(args.res_dir, args.ex_name))

    ckptdir = os.path.join(logdir, "checkpoints")

    monitor_metric = 'val_perp'
    filename = 'best-{epoch:02d}-{val_perp:.3f}'

    args.monitor_metric = monitor_metric

    callbacks.append(plc.ModelCheckpoint(
        monitor=monitor_metric,
        filename=filename,
        save_top_k=5,
        mode='min',
        save_last=True,
        dirpath=ckptdir,
        verbose=True,
        every_n_epochs=args.check_val_every_n_epoch,
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval=None))

    return callbacks


if __name__ == "__main__":
    # 🔍 optimization for NVIDIA-A100
    torch.set_float32_matmul_precision('medium')  # medium high highest

    args = parse_args()
    pl.seed_everything(args.seed)

    if args.ckpt_path is None:
        data_module = DInterface(**vars(args))
    else:
        data_module = DInterface.load_from_checkpoint(args.ckpt_path, **vars(args))

    data_module.setup('fit')
    gpu_count = torch.cuda.device_count()
    steps_per_epoch = math.ceil(len(data_module.trainset) / args.batch_size / gpu_count)
    callbacks = load_callbacks(args)

    if args.ckpt_path is None:
        model = MInterfaceMaskGIT(steps_per_epoch=steps_per_epoch, **vars(args))  # 🔍
    else:
        model = MInterfaceMaskGIT.load_from_checkpoint(args.ckpt_path, **vars(args))  # 🔍

    trainer_config = {
        'devices': -1,  # -1 Use all available GPUs
        'precision': 'bf16',  # Use 32-bit floating point precision
        # 'precision': '32',
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': 1,  # Number of nodes to use for distributed training
        "strategy": "ddp",  # ddp
        "accumulate_grad_batches": 1,
        'accelerator': 'cuda',  # cuda cpu
        'callbacks': load_callbacks(args),
        'logger': [
            plog.TensorBoardLogger(
                save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                name="tensorboard",
            ),  # 🔍
            plog.CSVLogger(args.res_dir, name=args.ex_name)],
        'gradient_clip_val': 0.5,
    }

    trainer = Trainer(**trainer_config)
    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)


