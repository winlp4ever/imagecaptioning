import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import Vis, Nlp
from store import cfg, model_urls
# q1: how to load weight partially
# q2: nlp with pytorch

class Nic(nn.Module):
    def __init__(self, vocab_size, embed_size=512):
        super(Nic, self).__init__()
        self.cnn = Vis(cfg['E'], embed_size, pretrained=True, link=model_urls['vgg19'])
        self.rnn = Nlp(vocab_size, embed_size)

    def forward(self, im, cap_enc):
        im_feats = self.cnn(im)
        return self.rnn(im_feats, cap_enc)[:, :,1 :-1]


class Nic_model(object):
    def __init__(self, *args, **kwargs):
        self.net = Nic(*args, **kwargs)
        self.loss = nn.NLLLoss()
        self.optim = optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-5)

    def to(self, device):
        self.net = self.net.to(device)

    def _train_ep(self, data_loader, device, epoch, args):
        self.net.train()
        self.net.cnn._fix_in_training()

        loss = 0

        for batch_idx, (im, cap_enc) in enumerate(data_loader):
            im, cap_enc = im.to(device), cap_enc.to(device)
            self.optim.zero_grad()
            l = self.loss(self.net(im, cap_enc), cap_enc[:, 1:])
            l.backward()
            self.optim.step()

            if batch_idx % args.log_interval == 0:
                print('ep {}: {:.0f}%({}/{})\tnllloss: {:.4f}'.format(epoch, batch_idx / len(data_loader) * 100,
                        batch_idx * len(im), len(data_loader.dataset), l), flush=True, end='\r')
            loss += l

        loss /= len(data_loader)

        if epoch % args.sv_interval == 0:
            self.save_checkpoint(args,
                                 {
                                     'epoch': epoch,
                                     # 'arch': args.arch,
                                     'state_dict': self.net.state_dict(),
                                     'optimizer': self.optim.state_dict(),
                                 }, epoch)

    def save_checkpoint(self, args, state, epoch):
        filename = os.path.join(args.ckpt_path, 'checkpoint-{}.pth.tar'.format(epoch))
        torch.save(state, filename)
