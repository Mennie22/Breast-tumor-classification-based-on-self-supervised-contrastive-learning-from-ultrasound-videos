import argparse

parser = argparse.ArgumentParser(description='DEMO')#其他参数百度

parser.add_argument('--model', type=str,default='Densenet169',help='Choose the type of model to train or test')
parser.add_argument('--load', type=bool,default=False, help='load_statedict or not')
parser.add_argument('--load_statedict', type=str,default='', help='load_statedict')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,help='number of GPUs')
parser.add_argument('--device', type=str, default='cuda:0',help='Id of GPUs')
parser.add_argument('--seed', type=int, default=1234,help='random seed')

# Data specifications
parser.add_argument('--pretrain_data_dir', type=str, default='../data/images/train_data', help='pretrain dataset directory')
parser.add_argument('--train_data_txt', type=str, default='../data/images/train.txt', help='train dataset txt')
parser.add_argument('--test_data_dir', type=str, default='../data/images/test_data', help='test dataset directory')
parser.add_argument('--test_data_txt', type=str, default='../data/images/test_txt', help='test dataset txt')
parser.add_argument('--no_augment', action='store_true', help='do not use data augmentation')

# Result(model or state or loss) to save
parser.add_argument('--save_models_dir', type= str,default='./model_statedict/', help='save all model statedicts')
parser.add_argument('--save_train_epoch_loss', type=str, default='result/loss/train_eval.xlsx', help='save the value of train and eval loss per epoch')
parser.add_argument('--save_eval_epoch_loss', type=str, default='result/loss/test.xlsx', help='save the value of test loss per epoch')
parser.add_argument('--tensorboard', type=bool, default=True, help='need tensorboard or not')
parser.add_argument('--dir_tensorboard_train', type=str, default='result/tensorboard_train/', help='the state is saved to here')
parser.add_argument('--dir_tensorboard_test', type=str, default='result/tensorboard_test/', help='the state is saved to here')
parser.add_argument('--save_fig', type=str, default='result/figs', help='all pics are saved to here')

# Model specifications
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--channels', type=int, default=1, help='number of color channels to use')

# Training specifications
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--margin', type=float, default=0.5, help='margin for hard triplet loss')
parser.add_argument('--num_topk_pos', type=int, default=3, help='positive pairs for hard triplet loss')
parser.add_argument('--num_topk_neg', type=int, default=12, help='negative pairs for hard triplet loss')
parser.add_argument('--crop_size', type=tuple, default=(224,224), help='crop size')
parser.add_argument('--choose_neg', type=int, default=88, help='choose negative')
parser.add_argument('--train_batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--train_shuffle', type=bool, default=True, help='shuffle for training')
parser.add_argument('--eval_batch_size', type=int, default=16, help='input batch size for eval')
parser.add_argument('--eval_shuffle', type=bool, default=False, help='shuffle for eval')
parser.add_argument('--test_batch_size', type=int, default=1, help='input batch size for training')
parser.add_argument('--test_shuffle', type=bool, default=False, help='shuffle for test')
'''训练阶段自定义画图'''
parser.add_argument('--train_draw_gap', type=int, default=200, help='how often to draw for training')
'''验证阶段tensorboard画图'''
parser.add_argument('--eval_draw_gap', type=int, default=100, help='how often to draw for eval')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Loss specifications
parser.add_argument('--loss_func', type=str, default='l2', help='choose the loss function: l2,l1,etc.')

# mode
parser.add_argument('--mode', type=str, default='train', choices=['train','test'], help='Choose to train or test or inference')



args, unparsed = parser.parse_known_args()
args1 = parser.parse_args()

''' vars()函数可以接受一个对象作为参数，然后返回该对象的属性和属性值组成的字典。
    如果没有传递参数给 vars() 函数，则默认返回当前作用域的局部变量的字典'''
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

