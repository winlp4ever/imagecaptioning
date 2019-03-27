import torch.nn as nn
from store import cfg, make_layers, model_urls
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn.utils.rnn import pack_padded_sequence

class ImgNN(nn.Module):
    def __init__(self, cfg, n_features, pretrained=True, link=None):
        super(ImgNN, self).__init__()
        self.features = make_layers(cfg, batch_norm=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.embeds = nn.Linear(4096, n_features)
        self.output = nn.Softmax()

        self._init_weights()

        if pretrained:
            assert link is not None
            pretrained_state = model_zoo.load_url(link)
            self.load_pretrained(pretrained_state)


    def load_pretrained(self, pretrained_state):
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
        x = self.classifier(x)
        x = self.embeds(x)
        x = self.output(x)
        return x

    def _fix_in_training(self):
        # call when training to fix this part of network untrained (already pretrained)
        for param in self.features.parameters():
            param.requires_grad = False


class Nlp(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Nlp, self).__init__()
        self.embeds = nn.Embedding(vocab_size, embed_size, scale_grad_by_freq=True)
        self.unit = nn.GRU(embed_size, embed_size, num_layers=1, batch_first=True)
        self.dense = nn.Linear(embed_size, vocab_size)
        self.out = nn.LogSoftmax(dim=2)

    def forward(self, img_embeds, captions, lengths, eval=False):
        embeddings = self.embeds(captions)
        features = torch.cat((img_embeds.unsqueeze(1), embeddings), 1)
        lengths = [1 + l for l in lengths]
        packed_features = pack_padded_sequence(features, lengths, batch_first=True)
        predicts, _ = self.unit(features)
        predicts = self.dense(predicts)
        return self.out(predicts).permute(0, 2, 1)


if __name__ == '__main__':
    cnn = ImgNN(cfg['E'], 512, True, link=model_urls['vgg19_bn'])
    print(cnn.classifier[0].weight.data)
