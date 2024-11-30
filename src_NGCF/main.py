# -*- coding: utf-8 -*-


import os
import torch
import shutil
import argparse
import numpy as np
from torch.optim import Adam
from trainers import MyTrainer
from models import PreFilter, Original
from utils import check_path, set_seed, get_graph_and_dataset, Timer
from datasets import UIGraphDataset, UIGraphDataset_neg, TestUserDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def show_args_info(args):
    print(f"--------------------Configure Info:--------------------")
    with open(args.log_file, 'a') as f:
        for arg in vars(args):
            info = f"{arg:<20} : {getattr(args, arg):>50}"
            print(info)
            f.write(info + '\n')


def main():
    parser = argparse.ArgumentParser()

    # system args
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--data_name', type=str, default='Yelp')
    parser.add_argument('--original_model', type=str, default='NGCF')
    parser.add_argument('--model_idx', type=int, default=1, help="model identifier 1,2,3,4,5,6,7...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--model_name", default='GFEraser', type=str)
    parser.add_argument("--seed", type=int, default=2023)

    # model args
    parser.add_argument("--embedding_size", type=int, default=64, help="the input and output embedding size")
    parser.add_argument("--graph_n_layers", type=int, default=3, help="number of LightGCN layers")

    # mutil-task args
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="weight of regularization task")
    parser.add_argument("--cl_weight", type=float, default=1.0, help="weight of cl task")
    parser.add_argument("--pos_bpr_weight", type=float, default=0.0175, help="weight of cl task")

    # training args
    parser.add_argument('--n_negatives', type=int, default=1, help="number of negative items")
    parser.add_argument('--epochs', type=int, default=20, help="the epoch to train each local model")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")

    # not studied
    parser.add_argument("--temp", type=float, default=1.0, help="temperature for InfoNCE")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay for adam")

    args = parser.parse_args()
    set_seed(args.seed)
    check_path(args.output_dir)

    assert args.data_name in ['ml-1m', 'Yelp', 'Amazon-Books', 'Beauty', 'Amazon-Electronics']
    if args.data_name == 'ml-1m':
        args.num_users = 6040
        args.num_items = 3706
    elif args.data_name == 'Yelp':
        args.num_users = 41772
        args.num_items = 30037
    elif args.data_name == 'Amazon-Books':
        args.num_users = 35736
        args.num_items = 38121
    elif args.data_name == 'Beauty':
        args.num_users = 22363
        args.num_items = 12101
    elif args.data_name == 'Amazon-Electronics':
        args.num_users = 147529
        args.num_items = 51700

    # check cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())

    # save model and args
    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.checkpoint_path = os.path.join(args.output_dir, args_str)
    if os.path.exists(args.checkpoint_path):
        shutil.rmtree(args.checkpoint_path, ignore_errors=True)
    os.mkdir(args.checkpoint_path)
    args.original_model_path = os.path.join('../pretrained', args.data_name + '_pretrained_' + args.original_model + '.pt')
    args.log_file = os.path.join(args.checkpoint_path, args_str + '.txt')
    args.test_log_file = os.path.join(args.checkpoint_path, args_str + '-test.txt')

    args.original_data_file = os.path.join(args.data_dir, args.data_name, args.data_name + '_train.txt')
    args.train_data_file = os.path.join(args.data_dir, args.data_name, args.data_name + '_train_new.txt')
    args.test_data_file = os.path.join(args.data_dir, args.data_name, args.data_name + '_test_new.txt')

    # A is the PC interactions from original train data, B is the PC interactions data from original test data
    # Both A and B used for unlearning evaluate, only A for negative graph training
    args.special_neg_data_A = os.path.join(args.data_dir, args.data_name, args.data_name + '_train_neg.txt')
    args.special_neg_data_B = os.path.join(args.data_dir, args.data_name, args.data_name + '_test_neg.txt')
    show_args_info(args)

    # load all data
    data_dict = get_graph_and_dataset(args)
    args.data_dict = data_dict

    # create u-i graph dataloader
    ui_dataset = UIGraphDataset(args)
    ui_sampler = RandomSampler(ui_dataset)
    ui_dataloader = DataLoader(ui_dataset, sampler=ui_sampler, batch_size=args.batch_size)
    ui_dataset_neg = UIGraphDataset_neg(args)
    ui_sampler_neg = RandomSampler(ui_dataset_neg)
    ui_dataloader_neg = DataLoader(ui_dataset_neg, sampler=ui_sampler_neg, batch_size=args.batch_size)

    # create test user dataloader
    args.device = torch.device("cuda" if args.cuda_condition else "cpu")
    user_list = list(range(0, args.num_users))
    test_user_dataset = TestUserDataset(args, user_list)
    test_user_sampler = SequentialSampler(test_user_dataset)
    test_user_dataloader = DataLoader(test_user_dataset, sampler=test_user_sampler, batch_size=args.batch_size,
                                      shuffle=False, drop_last=False)

    # Loading the original model for computing the initial ranking of PC interactions
    original_model = Original(args=args)
    original_model.load_state_dict(torch.load(args.original_model_path))
    if args.cuda_condition:
        original_model = original_model.to('cuda')
    args.user_emb, args.item_emb = original_model.get_emb()

    # initialize model and trainer
    model = PreFilter(args=args)
    # model.load_state_dict(torch.load(args.original_model_path), strict=False)
    optimizer = Adam(model.parameters(), weight_decay=args.weight_decay)
    if args.cuda_condition:
        model = model.to('cuda')

    trainer = MyTrainer(model, original_model, ui_dataloader, ui_dataloader_neg, test_user_dataloader, optimizer, args)
    # Evaluating the performance of the original model on the test set
    trainer.test_iteration(-1)

    timer = Timer()

    for epoch in range(1, args.epochs + 1):
        timer.start()
        trainer.train_iteration(epoch)
        timer.pause()
        if epoch == args.epochs or epoch == 1:
            trainer.test_iteration_get_neg_rank(epoch, test_user_dataloader)
        print('---------------Start Testing-------------------')
        result_list = trainer.test_iteration(epoch)

    total_elapsed_time = timer.get_elapsed_time()
    with open(args.test_log_file, 'a') as f:
        f.write(str('total training time: ' + str(total_elapsed_time / 60) + 'mins') + '\n')
    print('Finish training')


main()
