from dataset import load_vocab, MyCoco
from nltk.translate.bleu_score import sentence_bleu
from torchvision import datasets, transforms
import utils
import argparse
import torch
from model import Captor
from torch.nn.utils.rnn import pad_sequence
from play_with import beamsearch
import random


def _eval(model, im, cap_enc):
    model.net.eval()
    with torch.no_grad():
        im = im[None]
        length = [len(cap_enc)]
        cap_enc = cap_enc[None] # add dim batch
        prob = model.net(im, cap_enc, length)
    return prob


def main(args):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(random.randint(1, 10000))
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    words = load_vocab()
    vocab = {i: w for w, i in words.items()}

    coco = datasets.CocoCaptions(args.root_dir, args.anno_path)
    mycoco = MyCoco(words, args.root_dir, args.anno_path,
                   transform=transforms.Compose([
                        transforms.Resize([args.im_size] * 2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.407, 0.457, 0.485],  # subtract imagenet mean
                                      std=[1, 1, 1]),
                   ]))
    model = Captor(args.lr, args.weight_decay, args.lr_decay_rate, len(words), args.embed_size)
    model.to(device)

    model.load_checkpoint(args.ckpt_path)

    print('dataset length {}'.format(len(mycoco)))
    score = 0
    for i in range(len(mycoco)):
        im, cap_enc = mycoco[i]
        im_, caps = coco[i]
        pred = beamsearch(model, device, im, vocab, return_sentence=False)
        s = utils.bleu_score(utils.to_word_bags(caps), pred, n=args.bleu_n)
        score = (score * i + s) / (i + 1)
        print('processing {}th image... score: {:.2f}'.format(i, score), flush=True, end='\r')

    print('\navg bleu score: {}'.format(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--im-size', nargs='?', type=int, default=299)

    parser.add_argument('--embed-size', nargs='?', type=int, default=512)
    parser.add_argument('--seq-len', help='max sequence length', nargs='?', type=int, default=100)
    parser.add_argument('--lr-decay-interval', nargs='?', type=int, default=2000)
    parser.add_argument('--lr-decay-rate', nargs='?', type=float, default=1e-5)
    parser.add_argument('--epochs', nargs='?', type=int, default=100)
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3)
    parser.add_argument('--weight-decay', nargs='?', type=float, default=1e-2)
    parser.add_argument('--ckpt-path', nargs='?', default='./checkpoints')
    parser.add_argument('--root-dir', nargs='?', default='./data/val2014')
    parser.add_argument('--anno-path', nargs='?', default='./data/annotations/captions_val2014.json')

    parser.add_argument('--bleu-n', nargs='?', type=int, help='which BLEU-n score, options are: 1, 2, 3, 4.',
                        default=1)


    args = parser.parse_args()

    main(args)
