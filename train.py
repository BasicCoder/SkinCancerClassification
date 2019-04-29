import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import torchvision.models as models
from models.resnet_attn import ResNet50_Attn, ResNet50_Self_Attn
from dataset_input import SkinDataset, train_df, validation_df, test_df

import os
import shutil
import numpy as np
import tqdm

import argparse

parser = argparse.ArgumentParser(description='PyTorch Sketch Me That Shoe Example')
parser.add_argument('--net', type=str, default='resnet50', help='The model to be used (vgg16, resnet34, resnet50, resnet101, resnet152)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>,<save_latest_freq>+<epoch_count>...')
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')

parser.add_argument('--weight_decay', type=float, default=0.0005, help='Adm weight decay')

parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--classes', type=int, default=419,
                    help='number of classes')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='NetModel', type=str,
                    help='name of experiment')
parser.add_argument('--normalize_feature', default=False, type=bool,
                    help='normalize_feature')

best_acc = 0

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def train(train_loader, model, id_criterion, optimizer, epoch):
    model.train()
    for data_sample, y in train_loader:
        data_gpu = data_sample.cuda()
        y_gpu = y.cuda()

        output = model(data_gpu)

        err = id_criterion(output, y_gpu)
        err.backward()
        optimizer.step()

def test(train_loader,validation_set, model, id_criterion, epoch):
    model.eval()
    result_array = []
    gt_array = []
    for i in train_loader:
        data_sample, y = validation_set.__getitem__(i)
        data_gpu = data_sample.unsqueeze(0).cuda()
        output = model(data_gpu)
        result = torch.argmax(output)
        result_array.append(result.item())
        gt_array.append(y.item())

    correct_results = np.array(result_array) == np.array(gt_array)
    sum_correct = np.sum(correct_results)
    accuracy = sum_correct / train_loader.__len__()
    print('Epoch: {:d}  Prec@1: {:.10f}'.format(epoch, accuracy))
    return accuracy

def main():
    global args, best_acc
    args = parser.parse_args()
    opt = args
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}

    ###### DataSet ######
    composed = transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.CenterCrop(256),
                                   transforms.RandomCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    training_set = SkinDataset(train_df, transform=composed)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation_set = SkinDataset(validation_df, transform=composed)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_set = SkinDataset(validation_df, transform=composed)
    test_generator = torch.utils.data.SequentialSampler(test_set)

    ###### Model ######
    if opt.net == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=7)
    elif opt.net == 'resnet50_self_attn':
        model = ResNet50_Self_Attn(pretrained=True, out_features=7)
    elif opt.net == 'resnet50_attn':
        model = ResNet50_Attn(pretrained=True, out_features=7)

    if args.cuda:
        model.cuda()

    cudnn.benchmark = True

    ###### Criteria ######
    weights = [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2]
    class_weights = torch.FloatTensor(weights).cuda()
    id_criterion = nn.CrossEntropyLoss(weight=class_weights)
    schedulers = []
    optimizers = []
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    optimizers.append(optimizer)
    for optimizer in optimizers:
       schedulers.append(get_scheduler(optimizer, args))

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    for epoch in tqdm.tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):
        update_learning_rate(schedulers)
        # scheduler.step()
        # train for one epoch
        train(training_generator, model, id_criterion, optimizer, epoch)
        # evaluate on validation set
        if epoch % 5 == 0:
            prec1 = test(test_generator, validation_set, model, id_criterion, epoch)

            # remember best Accuracy and save checkpoint
            is_best = prec1 > best_acc
            best_acc = max(prec1, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec': best_acc,
            }, is_best)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 100))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def update_learning_rate(schedulers):
    for scheduler in schedulers:
        scheduler.step()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')

if __name__ == '__main__':
    main()