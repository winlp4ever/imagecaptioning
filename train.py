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
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                          std=[1, 1, 1]),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Nic_model(args.lr, args.weight_decay, args.lr_decay_rate, len(words), args.embed_size)
    model.to(device)

    init_epoch = model.load_checkpoint(args.ckpt_path)
    for i in range(1, args.epochs + 1):
        model._train_ep(train_loader, device, init_epoch + i, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--batch-size', nargs='?', type=int, default=128)
    parser.add_argument('--embed-size', nargs='?', type=int, default=512)
    parser.add_argument('--log-interval', nargs='?', type=int, default=1)
    parser.add_argument('--sv-interval', nargs='?', type=int, default=1)
    parser.add_argument('--lr-decay-interval', nargs='?', type=int, default=2000)
    parser.add_argument('--lr-decay-rate', nargs='?', type=float, default=1e-5)
    parser.add_argument('--epochs', nargs='?', type=int, default=100)
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3)
    parser.add_argument('--weight-decay', nargs='?', type=float, default=1e-2)
    parser.add_argument('--ckpt-path', nargs='?', default='./checkpoints')
    parser.add_argument('--root-dir', nargs='?', default='./data/train2014')
    parser.add_argument('--anno-path', nargs='?', default='./data/annotations/captions_train2014.json')

    parser.add_argument('--eval-dir', nargs='?', default='./data/val2014')
    parser.add_argument('--anno-eval', nargs='?', default='./data/annotations/captions_val2014.json')

    args = parser.parse_args()

    main(args)
