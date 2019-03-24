import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import Vis, Nlp
from store import cfg, model_urls
from tensorboardX import SummaryWriter
import os
import time
import glob


class Nic(nn.Module):
    def __init__(self, vocab_size, embed_size=512):
        super(Nic, self).__init__()
        self.cnn = Vis(cfg['E'], embed_size, pretrained=True, link=model_urls['vgg19'])
        self.rnn = Nlp(vocab_size, embed_size)

    def forward(self, im, cap_enc):
        im_feats = self.cnn(im)
        return self.rnn(im_feats, cap_enc)[:, :,1 :-1]


class Nic_model(object):
    def __init__(self, lr, weight_decay, lr_decay_rate, *args, **kwargs):
        self.net = Nic(*args, **kwargs)
        self.loss = nn.NLLLoss()
        self.optim = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.tracker = SummaryWriter()
        #self.init_lr = lr no needed for lr exponential decay strategy
        self.lr_decay_rate = lr_decay_rate

    def to(self, device):
        self.net = self.net.to(device)

    def _train_ep(self, data_loader, device, epoch, args):
        self.net.train()
        self.net.cnn._fix_in_training()
        loss = 0
        begin = time.time()
        for batch_idx, (im, cap_enc) in enumerate(data_loader):
            if (batch_idx + 1) % args.lr_decay_interval == 0:
                self._update_lr()
            im, cap_enc = im.to(device), cap_enc.to(device)
            self.optim.zero_grad()
            l = self.loss(self.net(im, cap_enc), cap_enc[:, 1:])
            l.backward()
            self.optim.step()
            if batch_idx % args.log_interval == 0:
                print('ep {}: {:.0f}%({}/{})\tnllloss: {:.4f} in {:.1f}s'.format(epoch, batch_idx / len(data_loader) * 100,
                        batch_idx * len(im), len(data_loader.dataset), l, time.time() - begin), flush=True, end='\r')
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
            for batch_idx, (im, cap_enc) in enumerate(data_loader):
                im, cap_enc = im.to(device), cap_enc.to(device)
                l = self.loss(self.net(im, cap_enc), cap_enc[:, 1:])
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
            checkpoint = torch.load(path)
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))
            return checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))
            return 0
