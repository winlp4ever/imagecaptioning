from model import Nic_model
import torch
from torchvision import transforms
from dataset import MyCoco, word_dict
import argparse


def main(args):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    root = './data/train2014'
    annot = './data/annotations/captions_train2014.json'
    words = word_dict()
    train_loader = torch.utils.data.DataLoader(
        MyCoco(words, 100, root, annot,
                       transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Nic_model(len(words), args.embed_size)
    model.to(device)
    model._train_ep(train_loader, device, 1, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--batch-size', nargs='?', type=int, default=64)
    parser.add_argument('--embed-size', nargs='?', type=int, default=512)
    parser.add_argument('--log-interval', nargs='?', type=int, default=1)
    parser.add_argument('--sv-interval', nargs='?', type=int, default=1)

    args = parser.parse_args()

    main(args)
