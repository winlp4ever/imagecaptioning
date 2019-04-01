import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
from dataset import load_vocab
import utils
import argparse
import torch
from model import Captor
import os
import random


def create_embeds(model, image):
    image = image[None]
    embeds = model.net.enc(image)
    embeds = embeds.view(embeds.size(0), -1)
    return model.net.embeds(embeds)

def beamsearch(model, device, image, vocab, return_sentence=True):
    model.net.eval()
    embeds = create_embeds(model, image.to(device))
    l = 0
    caps = []
    gen_cap = []
    count = 0
    cap_tens = None
    while True:
        predict = model.net.dec(embeds, cap_tens, [l])[:, :, -1]
        id = np.argmax(predict.cpu().data.numpy(), axis=1)[0]
        if id == 1 or count > 20:
            break
        gen_cap.append(vocab[id])
        caps.append(id)
        cap_tens = torch.Tensor([caps]).long().to(device)
        l += 1
        count += 1
    if return_sentence:
        return ' '.join(gen_cap[1:])
    return gen_cap[1:]

def main(args):
    preprocess = transforms.Compose([
         transforms.Resize([args.im_size] * 2),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.407, 0.457, 0.485],  # subtract imagenet mean
                       std=[1, 1, 1]),
    ])

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(random.randint(1, 10000))
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    words = load_vocab()
    vocab = {i: w for w, i in words.items()}

    model = Captor(args.lr, args.weight_decay, args.lr_decay_rate, len(words), args.embed_size)
    model.to(device)

    model.load_checkpoint(args.ckpt_path)

    im = Image.open(os.path.join('images', args.fn))
    im.show()
    img = preprocess(im)
    print("auto caption: {}.".format(beamsearch(model, device, img, vocab)))


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

    parser.add_argument('--fn', nargs='?', default='test.jpg')
    args = parser.parse_args()

    main(args)
