from model import Captor
import torch
from torchvision import transforms
from dataset import MyCoco, load_vocab, collate_fn
import argparse


def main(args):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'collate_fn': collate_fn, 'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    words = load_vocab()
    train_loader = torch.utils.data.DataLoader(
        MyCoco(words, args.root_dir, args.anno_path,
                       transform=transforms.Compose([
                            transforms.Resize([args.im_size] * 2),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.407, 0.457, 0.485],  # subtract imagenet mean
                                          std=[1, 1, 1]),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        MyCoco(words, args.eval_dir, args.anno_eval,
                transform=transforms.Compose([
                     transforms.Resize([args.im_size] * 2),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.407, 0.457, 0.485],  # subtract imagenet mean
                                   std=[1, 1, 1])])),
        batch_size=100, **kwargs)

    print(len(train_loader))

    model = Captor(args.lr, args.weight_decay, args.lr_decay_rate, len(words), args.embed_size)
    model.to(device)

    init_epoch = model.load_checkpoint(args.ckpt_path)
    for i in range(1, args.epochs + 1):
        model._train_ep(train_loader, device, init_epoch + i, args)
        if i % args.eval_interval == 0:
            model._eval_ep(test_loader, device, init_epoch + i, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--im-size', nargs='?', type=int, default=299)
    parser.add_argument('--batch-size', nargs='?', type=int, default=128)
    parser.add_argument('--embed-size', nargs='?', type=int, default=512)
    parser.add_argument('--log-interval', nargs='?', type=int, default=1)
    parser.add_argument('--sv-interval', nargs='?', type=int, default=1)
    parser.add_argument('--lr-decay-interval', nargs='?', type=int, default=2000)
    parser.add_argument('--lr-decay-rate', nargs='?', type=float, default=0.05)
    parser.add_argument('--epochs', nargs='?', type=int, default=100)
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3)
    parser.add_argument('--weight-decay', nargs='?', type=float, default=1e-5)
    parser.add_argument('--ckpt-path', nargs='?', default='./checkpoints')
    parser.add_argument('--root-dir', nargs='?', default='./data/train2014')
    parser.add_argument('--anno-path', nargs='?', default='./data/annotations/captions_train2014.json')

    parser.add_argument('--eval-dir', nargs='?', default='./data/val2014')
    parser.add_argument('--anno-eval', nargs='?', default='./data/annotations/captions_val2014.json')
    parser.add_argument('--eval-interval', nargs='?', type=int, default=3)

    args = parser.parse_args()

    main(args)
