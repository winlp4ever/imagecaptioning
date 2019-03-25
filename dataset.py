from torchvision import datasets, transforms
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence
import torch
import utils

def word_dict(coco):
    words = {}
    for i in range(len(coco)):
        print(i, flush=True, end='\r')
        _, captions = coco[i]
        for cap in captions:
            tokens = utils.token(cap)
            for tok in tokens:
                if tok not in words:
                    words[tok] = 1
                else:
                    words[tok] += 1
    vocab = []
    for w in words:
        if words[w] >= 5:
            vocab.append(w)

    vocab = ['*empty', '*begin', '*end', '*unk'] + vocab
    return vocab


def cap_to_idx(sen: str, words: dict):
    word_list = word_tokenize(sen)
    word_list = [w.lower() for w in word_list]
    word_list = ['*begin'] + word_list[:-1] + ['*end']
    #print(word_list)
    enc = []
    for w in word_list:
        if w in words:
            enc.append(words[w])
        else:
            enc.append(words['*unk'])
    return torch.Tensor(enc).long()


class MyCoco(datasets.CocoCaptions):
    def __init__(self, word_dict, *args, **kwargs):
        super(MyCoco, self).__init__(*args, **kwargs)
        self.word_dict = word_dict

    def __getitem__(self, idx):
        im, caps = super().__getitem__(idx)
        cap = caps[0]
        cap_enc = cap_to_idx(cap, self.word_dict)
        return im, cap_enc

def load_vocab(fn='data/vocab.txt', to_del={'.', '...', '-', ',', ';', ':', '!', '?'}):
    vocab = []
    with open(fn, 'r') as f:
        while True:
            line = f.readline()
            if line:
                tok = line.rstrip('\n')
                if tok not in to_del:
                    vocab.append(tok)
            else:
                break
    vocab = {w: i for i, w in enumerate(vocab)}
    return vocab

def collate_fn(data):
    # sort data by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge image tensors (stack)
    images = torch.stack(images, 0)

    # Merge captions
    #caption_lengths = [len(caption) for caption in captions]

    # zero-matrix num_captions x caption_max_length
    padded_captions = pad_sequence(captions, padding_value=0)
    return images, padded_captions#, caption_lengths


if __name__ == '__main__':
    root = './data/train2014'
    annot = './data/annotations/captions_train2014.json'

    coco = datasets.CocoCaptions(root, annot, transform=transforms.ToTensor())

    vocab = load_vocab('data/vocab.txt')
    print(vocab[':'])
    #print(vocab[5])
    print(len(coco))
    im, enc = coco[5]
    im = transforms.ToPILImage()(im)
    im.show()

    for i in range(len(coco)):
        _ = coco[i]
        print(i , flush=True, end='\r')
