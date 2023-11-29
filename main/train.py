import random

import math
import torch
from torch.utils.data import DataLoader, dataloader
import torch.nn as nn
import xlrd
from xlutils.copy import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
import utils
from utils import LoadDatasets,LOSS,draw
from parser_args import args


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def train(args):
    train_datasets,eval_datasets = LoadDatasets(args)
    train_loader = DataLoader(train_datasets,batch_size=args.train_batch_size, num_workers=args.n_threads, shuffle=args.train_shuffle)
    eval_loader = DataLoader(eval_datasets,batch_size=args.eval_batch_size, num_workers=args.n_threads, shuffle=args.eval_shuffle)

    loss = LOSS(args)

    cuda_n = 'cuda:'+str(args.GPU_id)
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
    evalloss = 0

    if args.load:
        print("=> loading checkpoint '{}'".format(args.load_statedict))
        checkpoint = torch.load(args.load_statedict, map_location=device)
        model.load_state_dict(checkpoint['model'])
        opt = checkpoint['opt']
        scheduler = checkpoint['scheduler']
        train_loader = checkpoint['train_loader']
        eval_loader = checkpoint['eval_loader']
        iter = checkpoint['iter']
        start_epoch = checkpoint['start_epoch']

        verbose = True
        trainloss = checkpoint['trainloss']
        evalloss = checkpoint['testloss']

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.dir_tensorboard_train)
    else:
        writer = None


    model.to(device)

    for epoch in range(start_epoch,epochs):
        if args.load and verbose:
            iter = iter
            verbose = False
        else:
            iter = 0

        model.train()
        for data in train_loader:
            iter += 1

            noise, label = data[0].to(device), data[1].to(device)
            gen = model.forward(noise)

            trainl = loss(gen, label)
            trainloss += trainl.item()

            print('train:  epoch:' + str(epoch + 1) + '_iter:' + str(iter), ' loss:', trainl.item())
            opt.zero_grad()
            trainl.backward()
            opt.step()

            '''图片可视化'''
            if iter % args.draw_gap == 0:
                draw(gen,label,epoch,iter,args)

        scheduler.step()

        if epoch % 1 == 0:
            state = {'model': model.state_dict(), 'opt': opt.state_dict(),'scheduler':scheduler,'train_loader':train_loader,
                     'eval_loader':eval_loader,'start_epoch': start_epoch,'iter': iter, 'trainloss': trainloss, 'evalloss': evalloss}

            torch.save(state, args.save_statedict + 'train' + '_' + str(epoch + 1) + '.pth')

        time=0
        model.eval()

        for data in eval_loader:
            time+=1

            noise, label = data[0].to(device), data[1].to(device)
            gen = model.forward(noise)

            evall = loss(gen, label)
            evalloss += evall.item()

            print('eval:  epoch:' + str(epoch + 1) + '_iter:' + str(iter), ' loss:', evall.item())

            if args.tensorboard and time % args.eval_draw_gap == 0:
                '''含有多个子图'''
                writer.add_images('eval pics', torch.stack(gen, label), global_step=time, dataformats='NCHW')

        if args.tensorboard:
            writer.add_scalars('train & eval loss by epoch',{"trainloss": trainloss/iter,
                                                            "evalloss": evalloss/time }, epoch+1)


        if epoch % 1 == 0:
            old = xlrd.open_workbook(args.save_train_epoch_loss)
            book = copy(old)
            sheettrain = book.get_sheet('train')
            r = old.sheets()[0].nrows
            sheettrain.write(r, 0, epoch + 1)
            sheettrain.write(r, 1, trainloss/iter)
            sheettrain.write(r, 2, evalloss/time)
            book.save(args.save_train_epoch_loss)
            trainloss = 0
            evalloss =0
def test(args):
    test_datasets,_ = LoadDatasets(args)
    test_loader = DataLoader(test_datasets, batch_size=args.test_batch_size, num_workers=args.n_threads, shuffle=args.test_shuffle)

    loss = LOSS(args)

    cuda_n = 'cuda:' + str(args.GPU_id)
    device = torch.device(cuda_n if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs
    model = utils.__dict__[args.model]()

    print("=> loading checkpoint '{}'".format(args.load_statedict))
    checkpoint = torch.load(args.load_statedict, map_location=device)
    model.load_state_dict(checkpoint['model'])

    iter = 0

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.dir_tensorboard_test)
    else:
        writer = None

    model.to(device)

    model.eval()
    for data in test_loader:
        iter += 1

        noise, label = data[0].to(device), data[1].to(device)
        gen = model.forward(noise)

        testloss = loss(gen, label).item

        print('test:  iter:' + str(iter), ' loss:', testloss)

        '''图片可视化'''

        if args.tensorboard:
            writer.add_scalar('test loss', testloss , iter)
            '''含有多个子图'''
            writer.add_images('test pics', torch.stack(gen, label), global_step=iter, dataformats='NCHW')


if __name__ == '__main__':
    if args.mode == 'train':
        print('start to train and eval....')
        train(args)

    elif args.mode == 'test':
        print('start to test....')
        test(args)


