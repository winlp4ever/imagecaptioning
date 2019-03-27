from dataset import load_vocab, MyCoco
from nltk.translate.bleu_score import sentence_bleu
from torchvision import datasets, transforms
import utils
import argparse
import torch
from model import Captor
from torch.nn.utils.rnn import pad_sequence


def _eval(model, im, cap_enc):
    model.net.eval()
    with torch.no_grad():
        im = im[None]
        cap_enc = cap_enc[None] # add dim batch
        prob = model.net(im, cap_enc)
    return prob


def main(args):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    words = load_vocab()
    idx_to_words = {i: w for w, i in words.items()}

    coco = datasets.CocoCaptions(args.root_dir, args.anno_path)
    mycoco = MyCoco(words, args.root_dir, args.anno_path,
                   transform=transforms.Compose([
                        transforms.Resize((224, 224)),
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
        prob = _eval(model, im.to(device), cap_enc.to(device))
        pred = utils.vec_to_words(prob, idx_to_words)
        im_, caps = coco[i]
        s = utils.bleu_score(utils.to_word_bags(caps), pred)
        print('processing {}th image... score: {:.2f}'.format(i, s), flush=True, end='\r')
        score += s

    score /= len(mycoco)
    print('\navg bleu score: {}'.format(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--embed-size', nargs='?', type=int, default=512)
    parser.add_argument('--seq-len', help='max sequence length', nargs='?', type=int, default=100)
    parser.add_argument('--lr-decay-interval', nargs='?', type=int, default=2000)
    parser.add_argument('--lr-decay-rate', nargs='?', type=float, default=1e-5)
    parser.add_argument('--epochs', nargs='?', type=int, default=100)
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3)
    parser.add_argument('--weight-decay', nargs='?', type=float, default=1e-2)
    parser.add_argument('--ckpt-path', nargs='?', default='./checkpoints')
    parser.add_argument('--root-dir', nargs='?', default='./data/train2014')
    parser.add_argument('--anno-path', nargs='?', default='./data/annotations/captions_train2014.json')


    args = parser.parse_args()

    main(args)
