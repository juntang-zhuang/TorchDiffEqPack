import argparse
import os, sys
import warnings
import pandas as pd
import time
import numpy as np
import yaml, csv
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as distributed
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import lib.layers as layers
import lib.utils as utils
import lib.odenvp as odenvp
from lib.datasets import CelebAHQ, Imagenet64

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import append_regularization_keys_header, append_regularization_csv_dict

import dist_utils
from dist_utils import env_world_size, env_rank
from torch.utils.data.distributed import DistributedSampler
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'adaptive_heun', 'bosh3', 'sym12async',
           'adalf', 'fixedstep_sym12async', 'fixedstep_adalf']


def get_parser():
    parser = argparse.ArgumentParser("Continuous Normalizing Flow")
    parser.add_argument("--datadir", default="~/Documents")
    parser.add_argument("--nworkers", type=int, default=4)
    parser.add_argument("--data", choices=["mnist", "svhn", "cifar10", 'lsun_church', 'celebahq', 'imagenet64'],
                        type=str, default="imagenet64")
    parser.add_argument("--dims", type=str, default="64,64,64")
    parser.add_argument("--strides", type=str, default="1,1,1,1")
    parser.add_argument("--num_blocks", type=int, default=2, help='Number of stacked CNFs.')

    parser.add_argument(
        "--layer_type", type=str, default="concat",
        choices=["ignore", "concat"]
    )
    parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
    parser.add_argument(
        "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu"]
    )
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--atol', type=float, default=1e-2, help='only for adaptive solvers')
    parser.add_argument('--rtol', type=float, default=1e-3, help='only for adaptive solvers')
    parser.add_argument('--step_size', type=float, default=0.1, help='only for fixed step size solvers')
    parser.add_argument('--first_step', type=float, default=0.166667, help='only for adaptive solvers')

    parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
    parser.add_argument('--test_atol', type=float, default=None)
    parser.add_argument('--test_rtol', type=float, default=None)
    parser.add_argument('--test_step_size', type=float, default=None)
    parser.add_argument('--test_first_step', type=float, default=None)

    parser.add_argument("--imagesize", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument('--time_length', type=float, default=1.0)
    parser.add_argument('--train_T', type=eval, default=False)
    parser.add_argument("--nrow", type=int, default=3)

    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--batch_size_schedule", type=str, default="",
        help="Increases the batchsize at every given epoch, dash separated."
    )
    parser.add_argument("--test_batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_iters", type=float, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
    parser.add_argument('--nbits', type=int, default=8)
    parser.add_argument('--div_samples', type=int, default=1)
    parser.add_argument('--squeeze_first', type=eval, default=False, choices=[True, False])
    parser.add_argument('--zero_last', type=eval, default=True, choices=[True, False])
    parser.add_argument('--seed', type=int, default=42)

    # Regularizations
    parser.add_argument('--reconstruction', type=float, default=0.0, help="|| x - decode(encode(x)) ||")
    parser.add_argument('--kinetic-energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--jacobian-norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total-deriv', type=float, default=None, help="int_t ||df/dt||^2")
    parser.add_argument('--directional-penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    parser.add_argument(
        "--max_grad_norm", type=float, default=10.0,
        help="Max norm of graidents"
    )

    parser.add_argument("--resume", type=str, default='experiments/imagenet64/example/best.pth', help='path to saved check point')
    parser.add_argument("--save", type=str, default="experiments/imagenet64/example")
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument('--validate', type=eval, default=False, choices=[True, False])

    parser.add_argument('--distributed', action='store_true', help='Run distributed training. Default True')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')

    # parser.add_argument('--skip-auto-shutdown', action='store_true',
    #                    help='Shutdown instance at the end of training or failure')
    # parser.add_argument('--auto-shutdown-success-delay-mins', default=10, type=int,
    #                    help='how long to wait until shutting down on success')
    # parser.add_argument('--auto-shutdown-failure-delay-mins', default=60, type=int,
    #                    help='how long to wait before shutting down on error')

    return parser


cudnn.benchmark = True
args = get_parser().parse_args()
# torch.manual_seed(args.seed)
nvals = 2 ** args.nbits

# Only want master rank logging
is_master = (not args.distributed) or (dist_utils.env_rank() == 0)
is_rank0 = args.local_rank == 0
write_log = is_rank0 and is_master


def add_noise(x, nbits=8):
    if nbits < 8:
        x = x // (2 ** (8 - nbits))
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
    else:
        noise = 1 / 2
    return x.add_(noise).div_(2 ** nbits)


def shift(x, nbits=8):
    if nbits < 8:
        x = x // (2 ** (8 - nbits))

    return x.add_(1 / 2).div_(2 ** nbits)


def unshift(x, nbits=8):
    return x.add_(-1 / (2 ** (nbits + 1)))


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size)])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root=args.datadir, train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root=args.datadir, train=False, transform=trans(im_size), download=True)
    elif args.data == "svhn":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.SVHN(root=args.datadir, split="train", transform=trans(im_size), download=True)
        test_set = dset.SVHN(root=args.datadir, split="test", transform=trans(im_size), download=True)
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(
            root=args.datadir, train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
            ]), download=True
        )
        test_set = dset.CIFAR10(root=args.datadir, train=False, transform=None, download=True)
    elif args.data == 'celebahq':
        im_dim = 3
        im_size = 256 if args.imagesize is None else args.imagesize
        ''' 
        train_set = CelebAHQ(
            train=True, root=args.datadir, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
            ])
        )
        test_set = CelebAHQ(
            train=False, root=args.datadir,  transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
            ])
        )
        '''
        train_set = CelebAHQ(train=True, root=args.datadir)
        test_set = CelebAHQ(train=False, root=args.datadir)
    elif args.data == 'imagenet64':
        im_dim = 3
        if args.imagesize != 64:
            args.imagesize = 64
        im_size = 64
        train_set = Imagenet64(train=True, root=args.datadir)
        test_set = Imagenet64(train=False, root=args.datadir)
    elif args.data == 'lsun_church':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.LSUN(
            'data', ['church_outdoor_train'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
            ])
        )
        test_set = dset.LSUN(
            'data', ['church_outdoor_val'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
            ])
        )
    data_shape = (im_dim, im_size, im_size)

    def fast_collate(batch):

        imgs = [img[0] for img in batch]
        targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
        w = imgs[0].size[0]
        h = imgs[0].size[1]

        tensor = torch.zeros((len(imgs), im_dim, im_size, im_size), dtype=torch.uint8)
        for i, img in enumerate(imgs):
            nump_array = np.asarray(img, dtype=np.uint8)
            tens = torch.from_numpy(nump_array)
            if (nump_array.ndim < 3):
                nump_array = np.expand_dims(nump_array, axis=-1)
            nump_array = np.rollaxis(nump_array, 2)
            tensor[i] += torch.from_numpy(nump_array)

        return tensor, targets

    train_sampler = (DistributedSampler(train_set,
                                        num_replicas=env_world_size(), rank=env_rank()) if args.distributed
                     else None)

    if not args.distributed:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.nworkers, pin_memory=True, collate_fn=fast_collate
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.nworkers, pin_memory=True, collate_fn=fast_collate
        )

    # import pdb
    # pdb.set_trace()

    test_sampler = (DistributedSampler(test_set,
                                       num_replicas=env_world_size(), rank=env_rank()) if args.distributed
                    else None)

    if not args.distributed:
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.nworkers, pin_memory=True, collate_fn=fast_collate
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=args.test_batch_size,
            num_workers=args.nworkers, pin_memory=True, sampler=test_sampler, collate_fn=fast_collate
        )
    return train_loader, test_loader, data_shape


def compute_bits_per_dim(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp, reg_states = model(x, zero)  # run model forward

    reg_states = tuple(torch.mean(rs) for rs in reg_states)

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(nvals)) / np.log(2)

    return bits_per_dim, (x, z), reg_states


def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    model = odenvp.ODENVP(
        (args.batch_size, *data_shape),
        n_blocks=args.num_blocks,
        intermediate_dims=hidden_dims,
        div_samples=args.div_samples,
        strides=strides,
        squeeze_first=args.squeeze_first,
        nonlinearity=args.nonlinearity,
        layer_type=args.layer_type,
        zero_last=args.zero_last,
        alpha=args.alpha,
        cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns},
    )

    return model


# if __name__ == "__main__":
def main():
    # os.system('shutdown -c')  # cancel previous shutdown command

    if write_log:
        utils.makedirs(args.save)
        logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

        logger.info(args)

        args_file_path = os.path.join(args.save, 'args.yaml')
        with open(args_file_path, 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    if args.distributed:
        if write_log: logger.info('Distributed initializing process group')
        torch.cuda.set_device(args.local_rank)
        distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                       world_size=dist_utils.env_world_size(), rank=env_rank())
        assert (dist_utils.env_world_size() == distributed.get_world_size())
        if write_log: logger.info("Distributed: success (%d/%d)" % (args.local_rank, distributed.get_world_size()))
        device = torch.device("cuda:%d" % torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    else:
        device = torch.cuda.current_device()  #

    # import pdb; pdb.set_trace()
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_loader, test_loader, data_shape = get_dataset(args)

    trainlog = os.path.join(args.save, 'training.csv')
    testlog = os.path.join(args.save, 'test.csv')

    traincolumns = ['itr', 'wall', 'itr_time', 'loss', 'bpd', 'fe', 'total_time', 'grad_norm']
    testcolumns = ['wall', 'epoch', 'eval_time', 'bpd', 'fe', 'total_time', 'transport_cost']

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns).cuda()
    if args.distributed: model = dist_utils.DDP(model,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank)

    traincolumns = append_regularization_keys_header(traincolumns, regularization_fns)

    if not args.resume and write_log:
        with open(trainlog, 'w') as f:
            csvlogger = csv.DictWriter(f, traincolumns)
            csvlogger.writeheader()
        with open(testlog, 'w') as f:
            csvlogger = csv.DictWriter(f, testcolumns)
            csvlogger.writeheader()

    set_cnf_options(args, model)

    if write_log: logger.info(model)
    if write_log: logger.info("Number of trainable parameters: {}".format(count_parameters(model)))
    if write_log: logger.info('Iters per train epoch: {}'.format(len(train_loader)))
    if write_log: logger.info('Iters per test: {}'.format(len(test_loader)))

    # optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9,
                              nesterov=False)

    # restore parameters
    # import pdb; pdb.set_trace()
    if args.resume is not None:
        # import pdb; pdb.set_trace()
        print('resume from checkpoint')
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    # For visualization.
    if write_log: fixed_z = cvt(torch.randn(min(args.test_batch_size, 100), *data_shape))

    if write_log:
        time_meter = utils.RunningAverageMeter(0.97)
        bpd_meter = utils.RunningAverageMeter(0.97)
        loss_meter = utils.RunningAverageMeter(0.97)
        steps_meter = utils.RunningAverageMeter(0.97)
        grad_meter = utils.RunningAverageMeter(0.97)
        tt_meter = utils.RunningAverageMeter(0.97)

    if not args.resume:
        best_loss = float("inf")
        itr = 0
        wall_clock = 0.
        begin_epoch = 1
        chkdir = args.save
        '''
    elif args.resume and args.validate:
        chkdir = os.path.dirname(args.resume)
        wall_clock = 0
        itr = 0
        best_loss = 0.0
        begin_epoch = 0
        '''
    else:
        chkdir = os.path.dirname(args.resume)
        filename = os.path.join(chkdir, 'test.csv')
        print(filename)
        tedf = pd.read_csv(os.path.join(chkdir, 'test.csv'))
        trdf = pd.read_csv(os.path.join(chkdir, 'training.csv'))
        # import pdb; pdb.set_trace()
        wall_clock = trdf['wall'].to_numpy()[-1]
        itr = trdf['itr'].to_numpy()[-1]
        best_loss = tedf['bpd'].min()
        begin_epoch = int(tedf['epoch'].to_numpy()[-1] + 1)  # not exactly correct

    if args.distributed:
        if write_log: logger.info('Syncing machines before training')
        dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())

    for epoch in range(begin_epoch, begin_epoch + 1):
        # compute test loss
        print('Evaluating')
        model.eval()
        if args.local_rank == 0:
            utils.makedirs(args.save)
            # import pdb; pdb.set_trace()
            if hasattr(model, 'module'):
                _state = model.module.state_dict()
            else:
                _state = model.state_dict()
            torch.save({
                "args": args,
                "state_dict": _state,  # model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "fixed_z": fixed_z.cpu()
            }, os.path.join(args.save, "checkpt_%d.pth" % epoch))

        # save real and generate with different temperatures
        fig_num = 64
        if True:  # args.save_real:
            for i, (x, y) in enumerate(test_loader):
                if i < 100:
                    pass
                elif i == 100:
                    real = x.size(0)
                else:
                    break
            if x.shape[0] > fig_num:
                x = x[:fig_num, ...]
            # import pdb; pdb.set_trace()
            fig_filename = os.path.join(chkdir, "real.jpg")
            save_image(x.float() / 255.0, fig_filename, nrow=8)

        if True:  # args.generate:
            print('\nGenerating images... ')
            fixed_z = cvt(torch.randn(fig_num, *data_shape))
            nb = int(np.ceil(np.sqrt(float(fixed_z.size(0)))))
            for t in [ 1.0, 0.99, 0.98, 0.97,0.96,0.95,0.93,0.92,0.90,0.85,0.8,0.75,0.7,0.65,0.6]:
                # visualize samples and density
                fig_filename = os.path.join(chkdir, "generated-T%g.jpg" % t)
                utils.makedirs(os.path.dirname(fig_filename))
                generated_samples = model(t * fixed_z, reverse=True)
                x = unshift(generated_samples[0].view(-1, *data_shape), 8)
                save_image(x, fig_filename, nrow=nb)

if __name__ == '__main__':
    try:
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore", category=UserWarning)
            main()
        # if not args.skip_auto_shutdown: os.system(f'sudo shutdown -h -P +{args.auto_shutdown_success_delay_mins}')
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        import traceback

        traceback.print_tb(exc_traceback, file=sys.stdout)
        # in case of exception, wait 2 hours before shutting down
        # if not args.skip_auto_shutdown: os.system(f'sudo shutdown -h -P +{args.auto_shutdown_failure_delay_mins}')
