import ast
import csv
import re
import os
import sys
import glob
import time
import random
import argparse
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import gdown

warnings.simplefilter("ignore")


class BatchState:
    """
    Holds batch-level state for aspect group tracking.
    """
    def __init__(self):
        self.prev_sent = []
        self.cnt_rev = 0
        self.cnt_asp_gr = 0
        self.tmp_last = 0
        self.is_tmp_last = False

    def reset(self):
        self.prev_sent = []
        self.cnt_rev = 0
        self.cnt_asp_gr = 0
        self.tmp_last = 0
        self.is_tmp_last = False





# ===========================
# Environment and Seed Setup
# ===========================

def set_seed(seed=42):
    """
    Set random seed for reproducibility across numpy, torch, and python random.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

def setup_environment():
    """
    Set environment variables for deterministic behavior and performance.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

# ===========================
# Data Utilities
# ===========================

def csv_reader(file):
    """
    Read and preprocess CSV data for ABSA.
    Returns a list of tuples: (sentence, aspects, aspect, sentiment)
    """
    df = pd.read_csv(file, header=None, names=['sent', 'num_asp', 'aspects', 'sentiment'])
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    data = []
    pat = re.compile("[\'\" ]+,")
    for _, row in df_shuffled.iterrows():
        sent = row['sent'].lower()
        nb_aspects = int(row['num_asp'])
        aspects = row['aspects'].replace(u"\\xc2\\xa0", u" ")
        aspects = re.sub(r"([0-9]+\.[0-9]+)", r"\1 ", aspects)
        aspects = [re.sub("[ ]+", " ", re.sub(r"'([^s]|$)", r"\1", re.sub("(^')|('$)", "", re.sub(r"(.){1}'s", r"\1 's", x.replace("\xc2\xa0", " ").replace("\\", " ").replace('[',"").replace("\"","").replace(']',"").strip().lower().replace("(", " -LRB- ").replace(")", " -RRB- "))))).strip() for x in pat.split(aspects)]
        sentiments = [x.strip().replace("'","").replace('[',"").replace("\"","").replace(']',"").lower() for x in row["sentiment"].split(",")]
        for i in range(nb_aspects):
            data.append((sent, aspects, aspects[i], [sentiments[i]]))
    return data

def shuffle_file(file):
    """
    Shuffle lines in a file (in-place).
    """
    with open(file, "r") as inp:
        lines = [line for line in inp]
    with open(file, "w") as out:
        for line in lines:
            out.write(line)

def read_asp_embeddings(file):
    """
    Read aspect embeddings from a file.
    Returns a list of lists of floats.
    """

    converted_data = []

    with open(file, "r") as f:
        # reader = csv.reader(f)
        lines = f.readlines()
        lines = [line.strip().strip("'\"") for line in lines if line.strip()]  # Remove empty lines
        for row in lines:
            actual_data = ast.literal_eval(row)  # Convert string to list
            converted_data.append(actual_data)

    return converted_data


# ===========================
# Embedding Utilities
# ===========================

def Glove(glove_dir):
    """
    Load GloVe embeddings from file.
    Returns a dictionary mapping word to embedding vector.
    """
    embeddings_index = {}
    glove_path = glove_dir  
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def index_word_embeddings(word_index, embeddings_index, embedding_dim):
    """
    Create embedding matrix for words in word_index using embeddings_index.
    """
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


REC_EMBEDDING_DIM = 0
FILES = []


setup_environment()
set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')
parser.add_argument('--batch-size', type=int, default=25, metavar='BS', help='batch size')
parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
parser.add_argument('--hops', type=int, default=10, metavar='H', help='number of hops')
parser.add_argument('--hidden-size', type=int, default=400, metavar='HS', help='hidden size')
parser.add_argument('--output-size', type=int, default=400, metavar='OS', help='output size')
parser.add_argument('--dropout-p', type=float, default=0.5, metavar='DO1', help='embedding dropout')
parser.add_argument('--dropout-lstm', type=float, default=0.1, metavar='DO2', help='lstm dropout')
parser.add_argument('--nb-words', type=int, default=500000000, metavar='NB', help='Number of words in the vocabulary')
parser.add_argument('--dataset', default='Restaurants', metavar='D', help='Laptop or Restaurants')
parser.add_argument('--embedding-file', default='glove.6B.300d.vec',  # '/home/cemrifki/Sentiment_Analysis/sentiment-recnn-rnn-ensemble-IARM/glove.6B.300d.vec', 
                    metavar='EMB', help='The path to the embedding file')
parser.add_argument('--recursive-module', type=str, default='dependency', metavar='RM',
                    choices=['dependency', 'constituency', 'baseline'], help='dependency or constituency')
parser.add_argument('--dependency-epochs', type=int, default=5, metavar='DE', help='number of epochs for the dependency model')
parser.add_argument('--constit-epochs', type=int, default=3, metavar='CE', help='number of epochs for the constituency model')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = "cuda" if args.cuda else "cpu"
print("Using device:", device)
ftype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


HIDDEN_DIM          = args.hidden_size
OUTPUT_DIM          = args.output_size
HOP_SIZE            = args.hops
BATCH_SIZE          = args.batch_size
NB_EPOCH            = args.epochs
nb_words            = 500000000
MAX_SEQUENCE_LENGTH = 77 if args.dataset=='Laptop' else 69
MAX_ASPECTS         = 13
MAX_LEN_ASPECT      = 5 if args.dataset=='Laptop' else 19
EMBEDDING_DIM       = 300
REC_EMBEDDING_DIM   = 50 if args.dataset[0:4].lower() == "rest" else 30 # 50
REC_MODULE          = args.recursive_module

if args.recursive_module == "baseline": REC_EMBEDDING_DIM = 0

FILES = files = ["2014_" + args.dataset + "_train.csv", "2014_" + args.dataset + "_test.csv"]

training_data = csv_reader(files[0])
test_data = csv_reader(files[1])


# ===========================
# Preprocessing Class
# ===========================


class PreProcessing:    
    """
    Handles tokenization, batching, and tensor preparation for ABSA.
    """
    def __init__(self, tr_data, te_data, tokenizer, batch_size, max_seq_len, max_aspects, embedding_dim, rec_embedding_dim, device, args):
        self.tag_to_ix = {"positive": 0, "negative": 1, "neutral": 2}
        self.tokenizer = tokenizer
        self.sents = list(zip(*tr_data))[0]
        self.sents1 = list(zip(*te_data))[0]
        self.labels = list(zip(*tr_data))[3]
        self.aspects = list(zip(*tr_data))[1]
        self.aspect = list(zip(*tr_data))[2]
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_aspects = max_aspects
        self.embedding_dim = embedding_dim
        self.rec_embedding_dim = rec_embedding_dim
        self.device = device
        self.args = args

    def prepare_sequence(self, seq, to_ix):
        return [to_ix[w] for w in seq]

    def keras_data_prepare(self, fit=True):
        if fit:
            self.tokenizer.fit_on_texts(self.sents + self.sents1)
        sequences = self.tokenizer.texts_to_sequences(self.sents)
        data = pad_sequences(sequences, maxlen=self.max_seq_len)
        return data

    def same_list(self, l1, l2):
        return l1 == l2

    def prepare_data(self, data, batch_id, word_embeddings, rec_embeddings=None, batch_state=None):
        if batch_state is None:
            batch_state = BatchState()

        aspect_sequence=[]
        limit = [batch_id*self.batch_size, (batch_id+1)*self.batch_size]
        for item in self.aspects[limit[0]:limit[1]]:
            temp=self.tokenizer.texts_to_sequences(item)
            aspect_sequence.append(temp)
        aspect_ = self.tokenizer.texts_to_sequences(list(self.aspect[limit[0]:limit[1]]))
        train_temp=[]
        j=0
        for datam in data[limit[0]:limit[1]]:
            train_temp.append([datam,aspect_sequence[j],aspect_[j],self.labels[limit[0]:limit[1]][j]])
            j=j+1
        training_data_x0=[]
        training_data_x1=[]
        training_data_y=[]
        attention_mat2 =[]
        attention_mat = []

        if self.args.recursive_module == "baseline":
            

            for item1 in train_temp:
                sent, aspects, aspect, sentiment = item1[0], item1[1], item1[2], item1[3]
                att = []
                for i in range(0,len(sent)):
                    if sent[i] == 0:
                        att.append(0)
                    else:
                        att.append(1)

                att_tensor = autograd.Variable(torch.FloatTensor(att) if not args.cuda else torch.cuda.FloatTensor(att),requires_grad=False)

                temp_mask_sent = att_tensor.view(att_tensor.size()[0],-1).expand(-1, 2*EMBEDDING_DIM)
                att_tensor = att_tensor.unsqueeze(0)
                tensor = torch.LongTensor(sent) if not args.cuda else torch.cuda.LongTensor(sent)
                sent1 = autograd.Variable(tensor)

                aspects1=[]
                for item in aspects:
                    temp = torch.LongTensor(item) if not args.cuda else torch.cuda.LongTensor(item)
                    temp = autograd.Variable(temp)
                    temp = word_embeddings(temp)
                    temp = torch.mean(temp,dim=0)
                    aspects1.append(temp)

                aspect = torch.LongTensor(aspect) if not args.cuda else torch.cuda.LongTensor(aspect)
                aspect = autograd.Variable(aspect)

                label=self.prepare_sequence(sentiment, self.tag_to_ix)

                embeds=word_embeddings(sent1)

                aspect1= word_embeddings(aspect)
                aspect1= torch.mean(aspect1,dim=0)
                aspect1 = aspect1.expand(len(sent),-1)

                sepr = []
                att2 = []
                for i in range(0,MAX_ASPECTS-len(aspects)):
                    sepr.append(autograd.Variable(torch.zeros((MAX_SEQUENCE_LENGTH,2*EMBEDDING_DIM)).type(ftype).unsqueeze(0)))
                    att2.append(0)

                for item in aspects1:
                    item = item.expand(len(sent),-1)
                    sepr.append(torch.mul(torch.cat([embeds,item],dim=1),temp_mask_sent).unsqueeze(0))
                    att2.append(1)

                aspect1 = torch.mul(torch.cat([embeds,aspect1],dim=1),temp_mask_sent)

                att2_tensor = autograd.Variable(torch.FloatTensor(att2) if not args.cuda else torch.cuda.FloatTensor(att2),requires_grad=False).unsqueeze(0)
                sepr_tensor=torch.cat(sepr,dim=0)
                sepr_tensor = sepr_tensor.unsqueeze(0)
                training_data_x0.append(sepr_tensor)
                training_data_x1.append(aspect1.unsqueeze(0))
                training_data_y.append(label)
                attention_mat2.append(att2_tensor)
                attention_mat.append(att_tensor)

        # Recursive module
        else:

            j = 0
            rev_sep = []
            for datam in data[limit[0]:limit[1]]:
                rev_sep.append(len(aspect_sequence[j]))
                j=j+1

            ind = 0

            for item1 in train_temp:
                sent, aspects, aspect, sentiment = item1[0], item1[1], item1[2], item1[3]

                rec_asp_embs_ = rec_embeddings[batch_state.cnt_rev]
                rec_asp_emb = rec_asp_embs_[batch_state.cnt_asp_gr]

                if batch_state.is_tmp_last:
                    rev_sep[0] = batch_state.tmp_last
                    batch_state.is_tmp_last = False
                    rev_sep_cnt = rev_sep[0] - 1
                else:
                    rev_sep_cnt = rev_sep[ind]
                    rev_sep_cnt -= 1

                if ind == len(rev_sep) - 1:
                    if rev_sep_cnt == 0:
                        batch_state.cnt_rev += 1
                        batch_state.cnt_asp_gr = 0
                        batch_state.is_tmp_last = False
                    else:
                        batch_state.cnt_asp_gr += 1
                        batch_state.tmp_last = rev_sep_cnt
                        batch_state.is_tmp_last = True
                else:
                    if rev_sep_cnt == 0:
                        batch_state.cnt_rev += 1
                        batch_state.cnt_asp_gr = 0
                    else:
                        batch_state.cnt_asp_gr += 1
                        rev_sep[ind + 1] = rev_sep_cnt

                ind += 1

                batch_state.prev_sent = sent

                prev_sent = sent
                sent_len = np.count_nonzero(np.array(sent))
                voc_len = len(sent)

                att = []
                for i in range(0,len(sent)):
                    if sent[i] == 0:
                        att.append(0)
                    else:
                        att.append(1)

                att_tensor = autograd.Variable(torch.FloatTensor(att) if not args.cuda else torch.cuda.FloatTensor(att),requires_grad=False)

                temp_mask_sent = att_tensor.view(att_tensor.size()[0],-1).expand(-1, 2 * EMBEDDING_DIM + REC_EMBEDDING_DIM)
                att_tensor = att_tensor.unsqueeze(0)
                tensor = torch.LongTensor(sent) if not args.cuda else torch.cuda.LongTensor(sent)
                sent1=autograd.Variable(tensor)

                aspects1=[]
                for item in aspects:
                    temp = torch.LongTensor(item) if not args.cuda else torch.cuda.LongTensor(item)
                    temp = autograd.Variable(temp)
                    temp = word_embeddings(temp)
                    temp = torch.mean(temp,dim=0)
                    aspects1.append(temp)

                aspect = torch.LongTensor(aspect) if not args.cuda else torch.cuda.LongTensor(aspect)
                aspect = autograd.Variable(aspect)

                label=self.prepare_sequence(sentiment, self.tag_to_ix)

                embeds=word_embeddings(sent1)

                aspect1= word_embeddings(aspect)
                aspect1= torch.mean(aspect1,dim=0).to(device) 

                rec_asp_emb_tens = torch.FloatTensor(rec_asp_emb).to(device) if args.cuda else torch.FloatTensor(rec_asp_emb)
                rec_len = len(rec_asp_emb_tens)

                aspect1 = torch.cat((aspect1, rec_asp_emb_tens), dim=0)

                aspect1 = aspect1.expand(len(sent),-1)


                non_match_words = torch.FloatTensor([[0.0] * rec_len] * (voc_len - sent_len))
                match_words = torch.FloatTensor([[1.0] * rec_len] * sent_len)
                rec_expansion = torch.cat((non_match_words, match_words), dim=0)            




                sepr = []
                att2 = []
                for i in range(0,MAX_ASPECTS-len(aspects)):
                    sepr.append(autograd.Variable(torch.zeros((MAX_SEQUENCE_LENGTH, 2 * EMBEDDING_DIM + REC_EMBEDDING_DIM)).type(ftype).unsqueeze(0)))
                    att2.append(0)

                cnt_all_asps = 0

                
                for item in aspects1: 
                    
                    it_rec_asp_emb = rec_asp_embs_[cnt_all_asps]
                    cnt_all_asps += 1
                    item = torch.cat((item, torch.FloatTensor(it_rec_asp_emb).to(device)), dim=0)

                    item = item.expand(len(sent),-1).data
                    emb_mask = torch.mul(torch.cat([embeds,item],dim=1),temp_mask_sent).unsqueeze(0)
                    
                    sepr.append(emb_mask)
                    att2.append(1)

                aspect1 = torch.mul(torch.cat([embeds,aspect1],dim=1),temp_mask_sent)

                att2_tensor = autograd.Variable(torch.FloatTensor(att2) if not args.cuda else torch.cuda.FloatTensor(att2),requires_grad=False).unsqueeze(0)
                sepr_tensor=torch.cat(sepr,dim=0)
                sepr_tensor = sepr_tensor.unsqueeze(0)
                training_data_x0.append(sepr_tensor)
                training_data_x1.append(aspect1.unsqueeze(0))
                training_data_y.append(label)
                attention_mat2.append(att2_tensor)
                attention_mat.append(att_tensor)

        att2_var = torch.cat(attention_mat2, dim=0)
        att_var = torch.cat(attention_mat, dim=0)
        res =  torch.cat(training_data_x0,dim=0), torch.cat(training_data_x1, dim=0), autograd.Variable(torch.LongTensor(to_categorical(training_data_y, 3)) if not args.cuda else torch.cuda.LongTensor(to_categorical(training_data_y,3))), att2_var, att_var, batch_state
        return res
  
# ===========================
# Model Definition
# ===========================

class AttnRNN(nn.Module):
    """
    Attention-based RNN for ABSA.
    """
    def __init__(self, hop_size, batch_size, input_size, sent_size, output_size,
                 dropout_p, dropout_lstm, max_length, rec_embedding_dim, device):
        super(AttnRNN, self).__init__()
        self.hop_size = hop_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.sent_size = sent_size
        self.dropout_p = dropout_p
        self.dropout_lstm = dropout_lstm
        self.max_length = max_length
        self.rec_embedding_dim = rec_embedding_dim
        self.device = device

        self.hidden_sentence_gru = self.init_hidden2(self.batch_size)
        self.hidden_aspect_gru = self.init_hidden(self.batch_size)
        self.hidden_aspect_write_gru = self.init_hidden(self.batch_size)
        self.sentence_gru = nn.GRU(2 * input_size + rec_embedding_dim, self.sent_size)
        self.aspect_gru = nn.GRU(self.sent_size, self.output_size)
        self.aspect_write_gru = nn.GRU(self.output_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_lstm)
        self.attn = nn.Linear(self.sent_size, 1)
        self.attn2 = nn.Linear(1, 1)
        self.affine = nn.Linear(self.output_size, 3)
        self.dimproj = nn.Linear(self.sent_size, self.output_size)

    def forward(self, sents, aspects, attention_mat1, attention_mat2, batch_size):
        sents=sents.permute(1,2,0,3) # -> (aspect, seq, batch, embed*2)
        outputs = []
        alphas=[]
        for sent_asp in sents:
            embedded = self.dropout(sent_asp)

            output, hidden_sentence_gru = self.sentence_gru(embedded, self.hidden_sentence_gru)
            temp_attention_mat1 = attention_mat1.view(attention_mat1.size()[0],attention_mat1.size()[1],1).expand(-1,-1,output.size()[2])
            output = torch.mul(output.permute(1,0,2),temp_attention_mat1)
            output = self.dropout2(output)
            attn_weights = F.softmax(
                self.attn(output.permute(1,0,2)), dim=0)
            masked_attn_weights = torch.mul(attn_weights.squeeze().permute(1,0),attention_mat1)
            _sums = masked_attn_weights.sum(-1).unsqueeze(1).expand(-1,masked_attn_weights.size()[1])
            attentions = masked_attn_weights.div(_sums).unsqueeze(1).permute(2,0,1)
            alphas.append(attentions.permute(1,2,0).unsqueeze(0))

            attn_applied = torch.bmm(attentions.permute(1,2,0),
                                 output).squeeze()
            output = torch.relu(attn_applied)
            outputs.append(output.unsqueeze(0))

        aspec_rep = torch.cat(outputs, dim=0)
        output, hidden_aspect_gru = self.aspect_gru(aspec_rep,self.hidden_aspect_gru)

        temp_attention_mat2 = attention_mat2.view(attention_mat2.size()[0],attention_mat2.size()[1],1).expand(-1,-1,output.size()[2])

        output = torch.mul(output.permute(1,0,2),temp_attention_mat2)
        output = self.dropout2(output)

        aspects = aspects.permute(1,0,2)
        outputa_,hida_ = self.sentence_gru(aspects,self.hidden_sentence_gru)
        temp_attention_mat3 = attention_mat1.view(attention_mat1.size()[0],attention_mat1.size()[1],1).expand(-1,-1,outputa_.size()[2])
        outputa_ = torch.mul(outputa_.permute(1,0,2),temp_attention_mat3)
        attn_weights_ = F.softmax(
                self.attn(outputa_.permute(1,0,2)), dim=0)
        masked_attn_weights_ = torch.mul(attn_weights_.squeeze().permute(1,0),attention_mat1)
        _sums_ = masked_attn_weights_.sum(-1).unsqueeze(1).expand(-1,masked_attn_weights_.size()[1])
        attentions_ = masked_attn_weights_.div(_sums_).unsqueeze(1).permute(2,0,1)
        attn_applied_ = torch.bmm(attentions_.permute(1,2,0),
                                 outputa_).squeeze()
        if self.sent_size == self.output_size:
                    asp_proj = attn_applied_.unsqueeze(1)
        else:
            asp_proj = self.dimproj(attn_applied_).unsqueeze(1)

        output=output.permute(0,2,1)

        betas = []
        for i in range(0,self.hop_size):
            match = torch.bmm(asp_proj,output).permute(2,0,1)


            attn_weights2 = F.softmax(
                    self.attn2(match), dim=0)
            self.hidden_aspect_write_gru=self.init_hidden(batch_size)
            output_w, hidden_aspect_write_gru = \
            self.aspect_write_gru(output.permute(2,0,1),self.hidden_aspect_write_gru)

            output_w = torch.mul(output_w.permute(1,0,2),temp_attention_mat2)
            output_w = self.dropout2(output_w)


            masked_attn_weights2 = torch.mul(attn_weights2.squeeze().permute(1,0),attention_mat2)
            _sums2 = masked_attn_weights2.sum(-1).unsqueeze(1).expand(-1,masked_attn_weights2.size()[1])
            attentions2 = masked_attn_weights2.div(_sums2).unsqueeze(1).permute(2,0,1)


            attn_applied = torch.bmm(attentions2.permute(1,2,0), output_w.permute(0,1,2)).squeeze()

            betas.append(attentions2.permute(1,2,0))


            query = asp_proj.view(asp_proj.size()[0],asp_proj.size()[2])
            final_output = torch.add(attn_applied, query)

            final_output = torch.relu(final_output)
            asp_proj = final_output.unsqueeze(1)
            output = output_w.permute(0,2,1)
        asp_proj = F.log_softmax(self.affine(asp_proj.squeeze()),dim=1)
        return asp_proj, betas, torch.cat(alphas,0)

    def init_hidden(self, batch_size):
        return autograd.Variable(torch.zeros(1, batch_size, self.output_size).type(torch.FloatTensor if self.device == "cpu" else torch.cuda.FloatTensor))

    def init_hidden2(self, batch_size):
        return autograd.Variable(torch.zeros(1, batch_size, self.sent_size).type(torch.FloatTensor if self.device == "cpu" else torch.cuda.FloatTensor))

# ===========================
# Training and Evaluation
# ===========================

def accuracy(preds, true):
    """
    Compute accuracy as a percentage.
    """
    return sum(1 for x, y in zip(preds, true) if x == y) / float(len(preds)) * 100.


def train_main(onea, args, training_data, test_data, tr_recurs_asp_embs, test_recurs_asp_embs, device, embedding_dim, rec_embedding_dim, max_seq_len, max_aspects):
    """
    Trains an attention-based RNN model for aspect-based sentiment analysis.
    This function initializes data preprocessing, embeddings, and the model architecture,
    then performs the main training loop over the specified number of epochs. After each epoch,
    it evaluates the model on the test set and prints training and evaluation metrics.
    Args:
        onea (list): Indices of test samples with a single aspect, used for separate accuracy reporting.
        args (Namespace): Configuration arguments containing hyperparameters such as batch size, learning rate, etc.
        training_data (list): List of training samples.
        test_data (list): List of test samples.
        tr_recurs_asp_embs (np.ndarray): Recursive aspect embeddings for the training set.
        test_recurs_asp_embs (np.ndarray): Recursive aspect embeddings for the test set.
        device (str): Device to use for computation ('cpu' or 'cuda').
        embedding_dim (int): Dimensionality of word embeddings.
        rec_embedding_dim (int): Dimensionality of recursive aspect embeddings.
        max_seq_len (int): Maximum sequence length for input sentences.
        max_aspects (int): Maximum number of aspects per sample.
    Returns:
        model (nn.Module): The trained attention-based RNN model.
        tokenizer (Tokenizer): The tokenizer fitted on the training data.
        word_embeddings (nn.Embedding): The embedding layer with pretrained weights.

    """
    tokenizer = Tokenizer(num_words=args.nb_words)
    prep = PreProcessing(training_data, test_data, tokenizer, args.batch_size, max_seq_len, max_aspects, embedding_dim, rec_embedding_dim, device, args)
    data = prep.keras_data_prepare()

    we = Glove(args.embedding_file)
    ei = index_word_embeddings(tokenizer.word_index, we, embedding_dim)
    word_embeddings = nn.Embedding(len(tokenizer.word_index) + 1, embedding_dim, padding_idx=0)
    word_embeddings.weight = nn.Parameter(torch.FloatTensor(ei) if device == "cpu" else torch.cuda.FloatTensor(ei))
    word_embeddings.weight.requires_grad = False

    model = AttnRNN(args.hops, args.batch_size, embedding_dim, args.hidden_size, args.output_size,
                    args.dropout_p, args.dropout_lstm, max_seq_len, rec_embedding_dim, device)
    if device == "cuda":
        model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters()) + [word_embeddings.weight]), lr=args.lr, weight_decay=args.l2)

    batch_count = int(np.ceil(len(training_data) / float(args.batch_size)))

    for i in tqdm(range(args.epochs)):
        start_time = time.time()
        loss_tot = []
        true_label=[]
        pred_res=[]
        model.train()
        batch_state = BatchState()
        for batch_id in range(batch_count):
            optimizer.zero_grad()
            bdata_x0, bdata_x1, bdata_y, attention_mat2, attention_mat1, batch_state = prep.prepare_data(
                data, batch_id, word_embeddings, tr_recurs_asp_embs, batch_state)
            model.hidden_sentence_gru = model.init_hidden2(bdata_x0.size()[0])
            model.hidden_aspect_gru = model.init_hidden(bdata_x0.size()[0])
            model.hidden_aspect_write_gru = model.init_hidden(bdata_x0.size()[0])
            
            prediction, _, _ = model(bdata_x0, bdata_x1, attention_mat1, attention_mat2, bdata_x0.size()[0])
            prediction = prediction.to(device)
            prediction = torch.where(torch.isnan(prediction),  torch.FloatTensor(1).uniform_(0.0, 0.1).to(device), prediction)
            loss = loss_function(prediction, torch.max(bdata_y, 1)[1])
            loss_tot.append(loss.item())
            pred_label = prediction.data.max(1)[1].cpu().numpy()
            pred_res += [x for x in pred_label]
            true_data = torch.max(bdata_y, 1)[1].cpu()
            true_label+= [x for x in true_data.data]
            loss.backward()
            optimizer.step()

        preds,true,test_loss = test(test_data, model, tokenizer,
                word_embeddings, test_recurs_asp_embs, loss_function, i, onea, args, device, prep)
        print('Epoch %d train_loss %.4f train_acc %.2f test_loss %.4f test_acc %.2f time %.2f' % (i+1, np.mean(loss_tot), accuracy(pred_res, true_label), test_loss, accuracy(preds,true), time.time()-start_time))
        mul = set(range(len(true)))-set(onea)
        print('single_aspect %.2f mul_aspect %.2f' % (accuracy([preds[idx] for idx in onea],[true[idx] for idx in onea]), accuracy([preds[idx] for idx in mul],[true[idx] for idx in mul])))

    return model, tokenizer, word_embeddings


def test(test_data, model, tokenizer, word_embeddings, test_recurs_asp_embs, loss_function, epoch, onea, args, device, prep):
    """
    Evaluates the given model on the provided test data and computes predictions, true labels, and average loss.

    Args:
        test_data (list or Dataset): The test dataset containing input samples.
        model (torch.nn.Module): The trained model to be evaluated.
        tokenizer (Tokenizer): Tokenizer used for preprocessing the input data.
        word_embeddings (np.ndarray or torch.Tensor): Pre-trained word embeddings matrix.
        test_recurs_asp_embs (np.ndarray or torch.Tensor): Recursive aspect embeddings for the test set.
        loss_function (callable): Loss function used to compute the evaluation loss.
        epoch (int): Current epoch number (used for logging or saving intermediate results).
        onea (Any): Additional argument, purpose depends on the broader context.

    Returns:
        tuple:
            pred_res (list): List of predicted labels for the test data.
            true_label (list): List of true labels corresponding to the test data.
            float: Mean loss over all test batches.

    Notes:
        - The function prepares the test data using a PreProcessing class, processes it in batches, and evaluates the model in evaluation mode.
        - It handles NaN predictions by replacing them with small random values.
        - Attention weights (betas and alphas) are collected but not returned.
        
    """
    prep_test = PreProcessing(test_data, training_data, tokenizer, args.batch_size, MAX_SEQUENCE_LENGTH, MAX_ASPECTS, EMBEDDING_DIM, REC_EMBEDDING_DIM, device, args)

    data = prep_test.keras_data_prepare(False)
    model.eval()
    true_label=[]
    loss_tot = []
    pred_res=[]
    batch_count = int(np.ceil(len(test_data)/float(args.batch_size)))
    betas = []
    alphas = []
    batch_state = BatchState()
    for batch_id in range(batch_count):
        bdata_x0, bdata_x1, bdata_y, attention_mat2, attention_mat1, batch_state = prep_test.prepare_data(
            data, batch_id, word_embeddings, test_recurs_asp_embs, batch_state)
        model.hidden_sentence_gru = model.init_hidden2(bdata_x0.size()[0])
        model.hidden_aspect_gru = model.init_hidden(bdata_x0.size()[0])
        model.hidden_aspect_write_gru = model.init_hidden(bdata_x0.size()[0])
        preds, beta , alpha = model(bdata_x0,bdata_x1, attention_mat1, attention_mat2, bdata_x0.size()[0])
        betas +=[dat.data.cpu().numpy() for dat in beta]
        alphas.append(alpha.data.cpu().numpy())
        preds = preds.to(device)
        preds = torch.where(torch.isnan(preds),  torch.FloatTensor(1).uniform_(0.0, 0.1).to(device), preds)
        loss = loss_function(preds, torch.max(bdata_y, 1)[1])
        loss_tot.append(loss.item())
        pred_label = preds.data.max(1)[1].cpu().numpy()
        pred_res += [x for x in pred_label]
        true_data = torch.max(bdata_y, 1)[1].cpu()
        true_label+= [x for x in true_data.data]
    return pred_res, true_label, np.mean(loss_tot)

def download_stanford_corenlp():

    nlp_core_files = os.path.join("constituency", "stanford-corenlp-4.5.8")  # You can also specify a different path if needed by downloading it 
    # from the Stanford CoreNLP website.

    # Check if the directory exists
    if not os.path.exists(nlp_core_files):
        import zipfile 
        # The URL of the Stanford CoreNLP library (for constituency parsing)
        drive_url_for_stanford_nlp = 'https://drive.google.com/uc?id=1-MC8frbFx-O8cW2c5uCzm2KqArLUJdNU'

        
        nlp_core_files_zip = nlp_core_files + ".zip"

        print(f"Downloading the Stanford NLP core files to the constituency folder.")
        gdown.download(drive_url_for_stanford_nlp, nlp_core_files_zip, quiet=False)

        
        def unzip_file(zip_path, extract_to='constituency'):
            """
            Unzips a .zip file to the given directory.
            
            Args:
                zip_path (str): Path to the .zip file.
                extract_to (str): Directory to extract files into.
            """
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                print(f'Extracted to {os.path.abspath(extract_to)}')
            
        unzip_file(nlp_core_files_zip, "constituency")
        # Remove the zip file
        os.remove(nlp_core_files_zip)

    else:
        print(f"Using the existing Stanford CoreNLP library in the constituency folder.")



# ===========================
# Main Entry Point
# ===========================

def main():

    """
    Main entry point for ABSA training and evaluation.
    """

    # The URL of the GloVe embedding file (i.e., glove.6B.300d.vec)
    drive_url_for_glove = 'https://drive.google.com/uc?id=1JikYQspoDIxlfhEdv2ry8QkWgZLMrFXZ'
    output_path = args.embedding_file 

    

    if os.path.isfile(args.embedding_file):
        print(f"Using existing embedding file: {args.embedding_file}")
    else:
        print(f"Downloading the GloVe embedding file to the path {args.embedding_file}.")
        gdown.download(drive_url_for_glove, output_path, quiet=False)


    
    # Recursive module logic (dependency/constituency/baseline)
    if args.recursive_module == "dependency":
        print("Dependency module selected.")        
        sys.path.append(os.path.join("dependency", "treehopper"))
        # Step 1: Import the dependency files generator module and train the model
        
        import dep_files_generator_lexicon
        import train 
        dep_files_generator_lexicon.main([deepcopy(training_data), test_data])

        # Step 2: Import the evaluate module
        import evaluate 

        train.main(parser)  # Call the main() method of train.py    


        # Base models directory
        base_dir = os.path.join("dependency", "treehopper", "models", "saved_model")

        # Get full paths to all subfolders in models/
        subfolders = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))]

        # Find the latest modified subfolder
        latest_folder = max(subfolders, key=os.path.getmtime)
        print(f"Latest folder: {latest_folder}")

        # Find all .pth files in that folder
        pth_files = glob.glob(os.path.join(latest_folder, '*.pth'))

        if not pth_files:
            print("No .pth files found in the latest folder.")
        else:
            # Get the latest .pth file by modification time
            latest_pth = max(pth_files, key=os.path.getmtime)
            print(f"Latest .pth file: {latest_pth}")

        # Step 4: Simulate command-line arguments
        sys.argv = ["evaluate.py", "--model_path", latest_pth]

        # Step 5: Call the main() method
        evaluate.main([deepcopy(training_data), test_data])  # Call the main() method of evaluate.py

        corresp_asp_emb_files = {"train": "2014_" + args.dataset + "_train_dim_" + str(REC_EMBEDDING_DIM) + "_" + args.recursive_module[:3] +"_asp_embs.csv", "test": "2014_" + args.dataset + "_test_dim_"  + str(REC_EMBEDDING_DIM) + "_" + args.recursive_module[:3] + "_asp_embs.csv"}


        tr_asp_emb_file = corresp_asp_emb_files["train"]
        tr_recurs_asp_embs = read_asp_embeddings(tr_asp_emb_file)


        test_asp_emb_file = corresp_asp_emb_files["test"]
        test_recurs_asp_embs = read_asp_embeddings(test_asp_emb_file)

    elif args.recursive_module == "constituency":
        import subprocess

        try:
            subprocess.run(['java', '-version'], check=True, capture_output=True)
            print("Java is installed.")
        except subprocess.CalledProcessError:
            print("Java is not installed or not in PATH.")
            sys.exit(1)

        # 1. Stanford CoreNLP library for constituency parsing
        download_stanford_corenlp()
        # Set the CLASSPATH environment variable to include the Stanford CoreNLP library
        os.environ['CLASSPATH'] = "constituency/stanford-corenlp-4.5.8/*"

        print("Constituency module selected.") 
        sys.path.append("constituency")
        import constit_parsing_aspect_gr_extr
        import RecNN
        # 2. Call constit_parsing_aspect_gr_extr.py
        print("Running constit_parsing_aspect_gr_extr.py and generating PennTree files...")
        
        constit_parsing_aspect_gr_extr.main(training_data, test_data)
        # 3. Call RecNN.py
        print("Running RecNN.py...")
        RecNN.main([training_data, test_data])

        corresp_asp_emb_files = {"train": "2014_" + args.dataset + "_train_dim_" + str(REC_EMBEDDING_DIM) + "_" + args.recursive_module[:3] +"_asp_embs.csv", "test": "2014_" + args.dataset + "_test_dim_"  + str(REC_EMBEDDING_DIM) + "_" + args.recursive_module[:3] + "_asp_embs.csv"}

        tr_asp_emb_file = corresp_asp_emb_files["train"]
        tr_recurs_asp_embs = read_asp_embeddings(tr_asp_emb_file)

        test_asp_emb_file = corresp_asp_emb_files["test"]
        test_recurs_asp_embs = read_asp_embeddings(test_asp_emb_file)

    elif args.recursive_module == "baseline":
                
        tr_recurs_asp_embs = None
        test_recurs_asp_embs = None
    else:
        print("Invalid recursive module specified. Please choose 'dependency' or 'constituency'.") 
        sys.exit(1)

    onea = [i for i, (s, a, aa, l) in enumerate(test_data) if len(a) == 1]
    tonea = [i for i, (s, a, aa, l) in enumerate(training_data) if len(a) == 1]

    model, tokenizer, word_embeddings = train_main(onea, args, training_data, test_data, 
                                                   tr_recurs_asp_embs, test_recurs_asp_embs, device, EMBEDDING_DIM, REC_EMBEDDING_DIM, 
                                                   MAX_SEQUENCE_LENGTH, MAX_ASPECTS)

if __name__ == '__main__':
    main()

# Two exemplary commands are as follows:
# python3 ABSA_emb_gpu_final_newarch3.py --recursive-module dependency --dependency-epochs 5 --lr 0.001 --l2 0.0001 --dataset Laptop --hops 3 --epochs 7 --hidden-size 400 --output-size 300 --dropout-p 0.1 --dropout-lstm 0.2
# python3 ABSA_emb_gpu_final_newarch3.py --recursive-module constituency --constit-epochs 3 --lr 0.001 --l2 0.0001 --dataset Laptop --hops 3 --epochs 1 --hidden-size 400 --output-size 300 --dropout-p 0.1 --dropout-lstm 0.2