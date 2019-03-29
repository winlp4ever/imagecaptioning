import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import ImgNN, Nlp
from store import cfg, model_urls
from tensorboardX import SummaryWriter
import os
import time
import glob
from torch.nn.utils.rnn import pack_padded_sequence
from inception import inception_v3


class CapNet(nn.Module):
    def __init__(self, vocab_size, embed_size=512):
        super(CapNet, self).__init__()
        self.enc = inception_v3(pretrained=True)
        self.relu = nn.ReLU(True)
        self.embeds = nn.Linear(1000, 512)
        #self.enc = ImgNN(cfg['E'], embed_size, pretrained=True, link=model_urls['vgg19_bn'])
        self.dec = Nlp(vocab_size, embed_size)

    def forward(self, imgs, caps, lens):
        embeds = self.enc(imgs)
        if self.training:
            embeds = embeds[0]
        embeds = self.relu(embeds)
        embeds = self.embeds(embeds)
        return self.dec(embeds, caps, lens)

    def _fix_in_training(self):
        # call when training to fix this part of network untrained (already pretrained)
        for param in self.enc.parameters():
            param.requires_grad = False


class Captor(object):
    def __init__(self, lr, weight_decay, lr_decay_rate, *args, **kwargs):
        self.net = CapNet(*args, **kwargs)
        self.loss = nn.NLLLoss()
        self.optim = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.tracker = SummaryWriter()
        #self.init_lr = lr no needed for lr exponential decay strategy
        self.lr_decay_rate = lr_decay_rate

    def to(self, device):
        self.net = self.net.to(device)

    def _train_ep(self, data_loader, device, epoch, args):
        self.net.train()
        self.net._fix_in_training()
        loss = 0
        begin = time.time()
        for batch_idx, (imgs, caps, lens) in enumerate(data_loader):
            if (batch_idx + 1) % args.lr_decay_interval == 0:
                self._update_lr()
            imgs, caps = imgs.to(device), caps.to(device)
            self.optim.zero_grad()
            l = self.loss(self.net(imgs, caps, lens)[:, :, :-1], caps)
            l.backward()
            self.optim.step()
            if batch_idx % args.log_interval == 0:
                print('ep {}: {:.0f}%({}/{})\tnllloss: {:.4f} in {:.1f}s'.format(epoch, batch_idx / len(data_loader) * 100,
                        batch_idx * len(imgs), len(data_loader.dataset), l, time.time() - begin), flush=True, end='\r')
            loss += l
            self.tracker.add_scalar('train_loss', l, global_step=(epoch - 1) * len(data_loader) + batch_idx)

        loss /= len(data_loader)
        self.tracker.add_scalar('average_train_loss', loss, global_step=epoch)

        if epoch % args.sv_interval == 0:
            self.save_checkpoint(args,
                                 {
                                     'epoch': epoch,
                                     'state_dict': self.net.state_dict(),
                                     'optimizer': self.optim.state_dict(),
                                 }, epoch)

    def _eval_ep(self, data_loader, device, epoch, args):
        self.net.eval()
        loss = 0
        begin = time.time()
        print('\nepoch evaluating...')
        with torch.no_grad():
            for batch_idx, (imgs, caps, lens) in enumerate(data_loader):
                imgs, caps = imgs.to(device), caps.to(device)
                l = self.loss(self.net(imgs, caps, lens)[:, :, :-1], caps)
                print('{:.1f}%\ttime-consuming: {:.1f}'.format(
                            batch_idx / len(data_loader) * 100., time.time() - begin),
                            flush=True, end='\r')
                loss += l
        loss /= len(data_loader)
        self.tracker.add_scalar('eval_loss', loss, global_step=epoch)
        print('\nresult: {:.4f}'.format(loss))

    def _update_lr(self):
        # exponential decay
        # lr = init_lr * (rate) ** nbiters
        for group in self.optim.param_groups:
            group['lr'] *= (1 - self.lr_decay_rate)

    def save_checkpoint(self, args, state, epoch):
        filename = os.path.join(args.ckpt_path, 'checkpoint-{}.pth.tar'.format(epoch))
        torch.save(state, filename)

    def load_checkpoint(self, ckpt_path):
        max_ep = 0
        path = ''
        for fp in glob.glob(os.path.join(ckpt_path, '*')):
            fn = os.path.basename(fp)
            fn_ = fn.replace('-', ' ')
            fn_ = fn_.replace('.', ' ')
            epoch = int(fn_.split()[1])
            if epoch > max_ep:
                path = fp
                max_ep = epoch

        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
<<<<<<< HEAD
            if not torch.cuda.is_available():
                checkpoint = torch.load(path, map_location='cpu')
            checkpoint = torch.load(path)
=======
            if torch.cuda.is_available():
                checkpoint = torch.load(path)
            else:
                checkpoint = torch.load(path, map_location='cpu')
>>>>>>> 78e76b5e77756fd167f440c6ff9e0594c7879244
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))
            return checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))
            return 0
