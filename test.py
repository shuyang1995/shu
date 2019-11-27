"""Run testing given a trained model."""

import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision

from dataset import FoodDataSet
from model import Model
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale

parser = argparse.ArgumentParser(
    description="Standard image-level testing")
parser.add_argument('--data-root', type=str)
parser.add_argument('--test-list', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', type=str)
parser.add_argument('--test-crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of workers for data loader.')
parser.add_argument('--gpus', nargs='+', type=int, default=None)

args = parser.parse_args()

# Update the total number of classes in the dataset
num_class = 52

def main():
    net = Model(num_class, base_model=args.arch)

    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.crop_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.crop_size, net.scale_size, is_mv=(args.representation == 'mv'))
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported, but got {}.".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
        FoodDataSet(
            args.data_root,
            img_list=args.test_list,
            transform=cropping,
            is_train=False,
            ),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()

    data_gen = enumerate(data_loader)

    total_num = len(data_loader.dataset)
    output = []

    def forward_img(data):
        """
        Args:
            data (Tensor): size [batch_size, c, h, w]

        Returns:
            scores (Tensor) : size [batch_size, num_class]

        """
        with torch.no_grad():
            input_var = torch.autograd.Variable(data, volatile=True)
            scores = net(input_var)
            scores = scores.view((-1, args.test_crops) + scores.size()[1:])
            scores = torch.mean(scores, dim=1)
            return scores.data.cpu().numpy().copy()

    proc_start_time = time.time()

    for i, (data, label) in data_gen:
        # data = [1, c, h ,w], label = [1]
        img_scores = forward_img(data)
        output.append((img_scores[0], label[0]))
        cnt_time = time.time() - proc_start_time
        if (i + 1) % 100 == 0:
            print('image {} done, total {}/{}, average {} sec/image'.format(i, i+1,
                                                                            total_num,
                                                                            float(cnt_time) / (i+1)))

    img_pred = [np.argmax(x[0]) for x in output]
    img_labels = [x[1] for x in output]

    print('Accuracy {:.02f}% ({})'.format(
        float(np.sum(np.array(img_pred) == np.array(img_labels))) / len(img_pred) * 100.0,
        len(img_pred)))

if __name__ == '__main__':
    main()
