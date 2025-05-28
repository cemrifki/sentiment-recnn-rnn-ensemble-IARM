from __future__ import print_function

import gc
import os

import torch
import torch.optim as optim
from torch import nn
from model.model import TreeLSTMSentiment
from model.sentiment_trainer import SentimentTrainer

from data.embeddings import load_embedding_model

import sys

sys.path.append("..")


def choose_optimizer(args, model):

    if args.optim =='adam':
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.dep_lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        return optim.Adagrad([
                {'params': model.parameters(), 'lr': args.dep_lr}
            ], lr=args.dep_lr, weight_decay=args.wd)


def train(train_dataset, dev_dataset, vocab, args):
    # Optionally reweight loss per class to the distribution of classes in
    # the public dataset

    weight = torch.Tensor([1/0.024, 1/0.820, 1/0.156]) if args.reweight else None
    criterion = nn.NLLLoss(weight=weight)

    embedding_model = load_embedding_model(args, vocab)

    model = TreeLSTMSentiment(args=args, criterion=criterion, embeddings=embedding_model, vocab=vocab)

    if args.cuda:
        model.cuda()
        criterion.cuda()

    optimizer = choose_optimizer(args,model)

    # Create trainer object for training and testing
    trainer = SentimentTrainer(args, model ,criterion, optimizer, embedding_model)
    experiment_dir = os.path.join(os.getcwd(), args.saved, "models_" + args.name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        open(experiment_dir+"/"+"config.txt", "w+").write(str(args))
    max_dev = 0
    max_dev_epoch = 0
    for epoch in range(args.dep_epochs):
        train_loss = trainer.train(train_dataset)
        dev_loss, dev_acc, _, _, root_tensors = trainer.test(dev_dataset)
        dev_acc = torch.mean(dev_acc)
        print('==> Train loss   : %f \t' % train_loss, end="")
        print('Epoch ', epoch, 'dev percentage ', dev_acc) 
        model_filename = experiment_dir + '/' +'model_' + str(epoch) + '.pth'
        torch.save(model, model_filename)
        if dev_acc > max_dev:
            max_dev = dev_acc
            max_dev_epoch = epoch
            max_model_filename = model_filename
        gc.collect()
    print('epoch ' + str(max_dev_epoch) + ' dev score of ' + str(max_dev))

    return max_dev_epoch, max_dev, max_model_filename


