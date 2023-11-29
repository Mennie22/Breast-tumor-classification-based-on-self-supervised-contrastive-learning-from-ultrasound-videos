import torch.nn as nn
def LOSS(args):
    if args.mode== 'train':
        return nn.MSELoss(reduction='sum')
    elif args.mode == 'test':
        return nn.MSELoss(reduction='sum')
    else:
        raise IndexError
