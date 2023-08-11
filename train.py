import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from utils.scheduler import CosineScheduler
from datasets.loader import PairLoader
from models.FCGAuNet import *
from ssimloss import *

# torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='FCGA_uNet_l', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='/home/student/File/Fwenxuan/data/', type=str,
                    help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-IN/', type=str, help='dataset name')
parser.add_argument('--data_type', default='dehaze', type=str, help='data type')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='2', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(train_loader, network, criterion, optimizer, scaler, num_train_step, epoch, epoch_step):
    losses = AverageMeter()
    torch.cuda.empty_cache()

    network.train()
    with tqdm(total=num_train_step, desc=f'Epoch {epoch + 1}/{epoch_step}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= num_train_step:
                break
            source_img = batch['source'].cuda()
            target_img = batch['target'].cuda()

            with autocast(args.no_autocast):
                output = network(source_img)
                loss = criterion(output, target_img)

            losses.update(loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(**{'loss': losses.avg,
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    return losses.avg


def valid(val_loader, network, num_train_step, epoch, epoch_step):
    PSNR = AverageMeter()
    torch.cuda.empty_cache()
    network.eval()
    with tqdm(total=num_train_step, desc=f'Epoch {epoch + 1}/{epoch_step}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= num_train_step:
                break
            source_img = batch['source'].cuda()
            target_img = batch['target'].cuda()
            with torch.no_grad():  # torch.no_grad() may cause warning
                H, W = source_img.shape[2:]
                # print(H, W)
                source_img = pad_img(source_img,
                                     network.module.patch_size if hasattr(network.module, 'patch_size') else 16)
                output = network(source_img).clamp_(-1, 1)
                # print(output.shape)
                output = output[:, :, :H, :W]
            mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
            psnr = 10 * torch.log10(1 / mse_loss).mean()
            PSNR.update(psnr.item(), source_img.size(0))
            pbar.set_postfix(**{'PSNR': PSNR.avg,
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    return PSNR.avg


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # max_split_size_mb = 1024  # 设置最大的分割尺寸，单位是MB
    # torch.cuda.set_per_process_memory_fraction(0.5, device=device.index, max_split_size_mb=max_split_size_mb)
    network = eval(args.model.replace('-', '_'))().to(device)
    print(network)
    network = nn.DataParallel(network)
    print(setting)
    criterion = nn.L1Loss()
    # criterion = L1_Charbonnier_loss()
    ssim_loss = SSIM(window_size=11)
    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'], weight_decay=0.01)
    else:
        raise Exception("ERROR: unsupported optimizer")
    # lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=setting['epochs'], value_min=setting['lr'] * 1e-2,
    #                                warmup_t=50, const_t=0)
    # wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay', t_max=300)  # seems not to work
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 2e-2)
    scaler = GradScaler()
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    print(dataset_dir)
    train_dataset = PairLoader(dataset_dir, 'train/train', 'train',
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    num_train = len(train_dataset)

    val_dataset = PairLoader(dataset_dir, 'test/test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)
    num_val = len(val_dataset)
    print(num_train)
    print(num_val)
    save_dir = os.path.join(args.save_dir, args.exp)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    num_train_step = num_train // setting['batch_size']
    num_val_step = num_val // setting['batch_size']
    if not os.path.exists(os.path.join(save_dir, args.model + '.pth')):
        print('==> Start training, current model name: ' + args.model)
        # print(network)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        best_psnr = 0
        for epoch in range(setting['epochs'] + 1):
            print("Start training")
            loss = train(train_loader, network, criterion, optimizer, scaler, num_train_step, epoch,
                         setting['epochs'] + 1)

            writer.add_scalar('train_loss', loss, epoch)
            scheduler.step()
            # lr_scheduler.step(epoch + 1)
            # wd_scheduler.step(epoch + 1)

            if epoch % setting['eval_freq'] == 0:
                print("Start validate")
                avg_psnr = valid(val_loader, network, num_val_step, epoch,
                                 setting['epochs'] + 1)
                writer.add_scalar('valid_psnr', avg_psnr, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + '.pth'))

                writer.add_scalar('best_psnr', best_psnr, epoch)

        else:
            print('==> Existing trained model')
            exit(1)
