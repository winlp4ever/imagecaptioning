import torch.nn as nn
from store import cfg, make_layers, model_urls
import torch.utils.model_zoo as model_zoo
import torch


class Vis(nn.Module):
    def __init__(self, cfg, n_features, pretrained=True, link=None):
        super(Vis, self).__init__()
        self.features = make_layers(cfg, batch_norm=False)
        self.dense = nn.Linear(512 * 7 * 7, n_features)
        self.feats = nn.Softmax()

        self._init_weights()

        if pretrained:
            assert link is not None
            pretrained_state = model_zoo.load_url(link)
            state = self.state_dict()
            pretrained = { k:v for k,v in pretrained_state.items() if k in state and v.size() == state[k].size() }
            state.update(pretrained)
            self.load_state_dict(state)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = self.feats(x)
        return x[None] # add dim 0, new size = (1, batch_size, n_features)

    def _fix_in_training(self):
        # call when training to fix this part of network untrained (already pretrained)
        for param in self.features.parameters():
            param.requires_grad = False


class Nlp(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Nlp, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, max_norm=1)
        self.recur = nn.LSTM(embed_size, embed_size, num_layers=2, dropout=0.2)
        self.dense = nn.Linear(embed_size, vocab_size)
        self.logproba = nn.LogSoftmax(dim=2)

    def forward(self, im_feats, seq):
        len_seq = seq.shape[0]
        word_feats = self.embed(seq)
        feats = torch.cat((im_feats, word_feats.permute(1, 0, 2)))
        guess, _ = self.recur(feats)
        guess = self.dense(guess)
        return self.logproba(guess).permute(1, 2, 0)


if __name__ == '__main__':
    cnn = Vis(cfg['E'], 512, True, link=model_urls['vgg19'])
    print(cnn.features[2].weight.data)
