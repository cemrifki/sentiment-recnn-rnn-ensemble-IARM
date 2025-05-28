import argparse
from datetime import datetime
import re
import os
import sys

import torch

sys.path.append(os.path.join("..", ".."))

from ABSA_emb_gpu_final_newarch3 import  REC_EMBEDDING_DIM, args as main_args


def parse_args_train(parser = None):
        if parser == None:
            parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analysis Trees')
        parser.add_argument('--name',
                            default='{date:%Y%m%d_%H%M}'.format(
                                date=datetime.now()),
                            help='name for log and saved models')
        parser.add_argument('--saved', default='dependency/treehopper/models/saved_model',
                            help='name for log and saved models')
        parser.add_argument('--data', default='dependency/treehopper/training-treebank',
                            help='path to dataset')
        
        parser.add_argument('--emb_dir', default=os.path.dirname(main_args.embedding_file),
                            help='directory with embeddings')
        parser.add_argument('--emb_file', default=os.path.splitext(os.path.basename(main_args.embedding_file))[0],  # default='glove.840B.300',
                            help='file with embeddings')
        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--dep-epochs', default=main_args.dependency_epochs, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--mem_dim', default=REC_EMBEDDING_DIM, type=int,
                            help='size of LSTM hidden state')
        parser.add_argument('--recurrent_dropout_c', default=0.15, type=float,
                            help='probability of recurrent dropout for cell state')
        parser.add_argument('--recurrent_dropout_h', default=0.15, type=float,
                            help='probability of recurrent dropout for hidden state')
        parser.add_argument('--zoneout_choose_child', default=False, type=bool,
                            help='tba')
        parser.add_argument('--common_mask', default=False, type=bool,
                            help='tba')
        parser.add_argument('--dep-lr', default=0.05, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--emblr', default=0.1, type=float,
                            metavar='EMLR', help='initial embedding learning rate')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--reg', default=1e-4, type=float,
                            help='l2 regularization (default: 1e-4)')
        parser.add_argument('--optim', default='adagrad',
                            help='optimizer (default: adagrad)')
        parser.add_argument('--seed', default=123, type=int,
                            help='random seed (default: 123)')
        parser.add_argument('--reweight', default=False, type=bool,
                            help='reweight loss per class to the distrubition '
                                 'of classess in the public dataset')
        parser.add_argument('--split', default=0.1, type=float,
                            help='Train/val split size')
        parser.add_argument('--use_full_training_set', default=False,
                            help='Train of full Eval training set, '
                                 'i.e. train+dev')

        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        parser.set_defaults(cuda=False, train=False)

        args = parser.parse_args()
        return args



def set_arguments(parser = None, grid_args = None):
    # Convert dict to Namespace
    args = parse_args_train(parser)
    if grid_args !=None:
        if "embeddings" in grid_args:
            args.emb_dir = grid_args["embeddings"][0]
            args.emb_file = grid_args["embeddings"][1]
        for key, val in grid_args.items():
            setattr(args,key,val)
    args.calculate_new_words = True

    embedding_dim = r"(\d+)[dD]*$"


    dim_from_file = re.search(embedding_dim, args.emb_file)
    args.input_dim = int(dim_from_file.group(1)) if dim_from_file else 300
    args.num_classes = 3  # -1 0 1

    args.cuda = args.cuda and torch.cuda.is_available()

    args.test = args.use_full_training_set
    return args
