import os
from pretrain.main import args
import torch.nn as nn
import torch
from pretrain.utils import LoadDatasets
from torch.utils.data import DataLoader
import xlrd
from xlutils.copy import copy
from pretrain.utils import HardTripletloss,get_indices
from pretrain import utils

'''
    obtain indices
'''
indices = get_indices()


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def train(model):
    train_datasets = LoadDatasets(indices)
    train_loader = DataLoader(train_datasets, batch_size=args.train_batch_size, num_workers=args.n_threads, shuffle=args.train_shuffle)

    contrastive_loss = HardTripletloss(args)

    cuda_n = 'cuda:' + str(args.GPU_id)
    device = torch.device(cuda_n if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs
    model = utils.__dict__[args.model]()

    model.apply(init_weights)
    opt = torch.optim.Adam(model.parameters(), lr=9e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader), eta_min=1e-5)
    iter = 0
    start_epoch = 0
    verbose = False
    trainloss = 0

    if args.load:
        print("=> loading checkpoint '{}'".format(args.load_statedict))
        checkpoint = torch.load(args.load_statedict, map_location=device)
        model.load_state_dict(checkpoint['model'])
        opt = checkpoint['opt']
        scheduler = checkpoint['scheduler']
        train_loader = checkpoint['train_loader']
        iter = checkpoint['iter']
        start_epoch = checkpoint['start_epoch']

        verbose = True
        trainloss = checkpoint['trainloss']

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.dir_tensorboard_train)
    else:
        writer = None

    model.to(device)

    for epoch in range(start_epoch, epochs):
        if args.load and verbose:
            iter = iter
            verbose = False
        else:
            iter = 0

        model.train()
        for data in train_loader:
            iter += 1

            img = data[0].squeeze(dim=0).to(device)
            fs = model(img)
            closs = contrastive_loss(fs)
            loss = closs(fs)
            trainloss+=loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        scheduler.step()

        if epoch % 1 == 0:
            state = {'model': model.state_dict(), 'opt': opt.state_dict(), 'scheduler': scheduler,
                     'train_loader': train_loader, 'start_epoch': start_epoch, 'iter': iter, 'trainloss': trainloss}

            torch.save(state, args.save_statedict + 'train' + '_' + str(epoch + 1) + '.pth')

        if args.tensorboard:
            writer.add_scalar('train loss by epoch',trainloss / iter, epoch + 1)

        if epoch % 1 == 0:
            old = xlrd.open_workbook(args.save_train_epoch_loss)
            book = copy(old)
            sheettrain = book.get_sheet('train')
            r = old.sheets()[0].nrows
            sheettrain.write(r, 0, epoch + 1)
            sheettrain.write(r, 1, trainloss / iter)
            book.save(args.save_train_epoch_loss)
            trainloss = 0


if __name__ == '__main__':
    print('start to train...')
    train(args)



