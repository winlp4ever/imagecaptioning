from torchvision import datasets, transforms
from nltk import word_tokenize
import torch


def word_dict(fn='./data/10000.txt'):
    words = []
    with open(fn, 'r') as f:
        while True:
            line = f.readline()
            if line:
                words.append(line.rstrip('\n'))
            else:
                break
    words = ['*begin', '*end', '*unk', '-'] + words
    words = {w: i for i, w in enumerate(words)}
    return words


def cap_to_idx(sen: str, words: dict, seq_len=40):
    word_list = word_tokenize(sen)
    word_list = [w.lower() for w in word_list]
    assert seq_len >= len(word_list) + 1, sen
    word_list = ['*begin'] + word_list[:-1] + ['*end'] + ['-'] * (seq_len - len(word_list) - 1)
    #print(word_list)
    enc = []
    for w in word_list:
        if w in words:
            enc.append(words[w])
        else:
            enc.append(words['*unk'])
    return torch.Tensor(enc).long()


class MyCoco(datasets.CocoCaptions):
    def __init__(self, word_dict, seq_len, *args, **kwargs):
        super(MyCoco, self).__init__(*args, **kwargs)
        self.word_dict = word_dict
        self.seq_len = seq_len

    def __getitem__(self, idx):
        im, caps = super().__getitem__(idx)
        cap = caps[0]
        cap_enc = cap_to_idx(cap, self.word_dict, self.seq_len)
        return im, cap_enc


if __name__ == '__main__':
    root = './data/train2014'
    annot = './data/annotations/captions_train2014.json'
    words = word_dict()
    print(len(words))
    print(words)
    coco = MyCoco(words, 100, root, annot, transform=transforms.ToTensor())
    print(len(coco))
    im, enc = coco[5]
    im = transforms.ToPILImage()(im)
    im.show()
    print(enc)

    for i in range(len(coco)):
        _, enc = coco[i]
        print(i , flush=True, end='\r')
