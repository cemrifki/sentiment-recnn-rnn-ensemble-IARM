import os
import numpy as np


import os
import subprocess

# from gensim.models import FastText
import gensim
from gensim.models import KeyedVectors

import torch
from torch.nn import Embedding

from data.vocab import Vocab
import sys

sys.path.append("..")
sys.path.append(os.path.join("..", ".."))


def load_word_vectors(embeddings_path):
    print(embeddings_path)
    if os.path.isfile(embeddings_path + '.pth') and \
            os.path.isfile(embeddings_path + '.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(embeddings_path + '.pth')
        vocab = Vocab(filename=embeddings_path + '.vocab')
        return vocab, vectors
    if os.path.isfile(embeddings_path + '.model'):
        model = KeyedVectors.load(embeddings_path + ".model")
    if os.path.isfile(embeddings_path + '.vec'):
        model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path + '.vec', binary=False, no_header=True)
    list_of_tokens = model.key_to_index.keys()

    # print( '==> Vocabulary size: %d ' % len(list_of_tokens))
    vectors = torch.zeros(len(list_of_tokens), model.vector_size)
    with open(embeddings_path + '.vocab', 'w', encoding='utf-8') as f:
        for token in list_of_tokens:
            f.write(token+'\n')
    vocab = Vocab(filename=embeddings_path + '.vocab')
    for index, word in enumerate(list_of_tokens):
        vectors[index, :] = torch.from_numpy(model[word])
    return vocab, vectors


def apply_not_known_words(emb,args, not_known,vocab):
    new_words = 'dependency/treehopper/tmp/new_words.txt'
    f = open(new_words, 'w', encoding='utf-8')
    for item in not_known:
        f.write("%s\n" % item)
    cmd = " ".join(["./fastText/fasttext", "print-word-vectors",
                    args.emb_dir + "/" + args.emb_file + ".bin", "<", new_words])
    print(cmd)
    ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    new_words_embeddings = [x.split(" ")[:-1] for x in output.decode("utf-8").split("\n")]
    for word in new_words_embeddings:
        if args.input_dim == len(word[1:]):
            emb[vocab.get_index(word[0])] = torch.from_numpy(np.asarray(list(map(float, word[1:]))))
        else:
            print('Word embedding from subproccess has different length than expected')
    return emb


def load_embedding_model(args, vocab):
    embedding_model = Embedding(vocab.size(), args.input_dim)

    if args.cuda:
        embedding_model = embedding_model.cuda()
    emb_file = os.path.join(args.data, args.emb_file + '_' + args.dataset + '_emb.pth')
    
    if os.path.isfile(emb_file) and torch.load(emb_file).size()[1] == args.input_dim and torch.load(emb_file).size()[0] == vocab.size():
        print('==> File found, loading to memory')
        emb = torch.load(emb_file)
    else:
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.emb_dir, args.emb_file))
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        print(vocab.size())
        emb = torch.zeros(vocab.size(), glove_emb.size(1))
        not_known = []
        for word in vocab.token_to_idx.keys():
            if glove_vocab.get_index(word):
                emb[vocab.get_index(word)] = glove_emb[glove_vocab.get_index(word)]
            else:
                not_known.append(word)
                emb[vocab.get_index(word)] = torch.Tensor(emb[vocab.get_index(word)].size()).normal_(-0.05, 0.05)
        if args.calculate_new_words:
            emb = apply_not_known_words(emb, args, not_known, vocab)

        torch.save(emb, emb_file)

    if args.cuda:
        emb = emb.cuda()
    # plug these into embedding matrix inside model
    embedding_model.state_dict()['weight'].copy_(emb)
    return embedding_model

