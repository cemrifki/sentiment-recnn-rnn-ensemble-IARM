
"""
RecNN.py

A PyTorch implementation of a Recursive Neural Tensor Network (RNTN) for constituency parsing and aspect-based sentiment analysis.

This module provides:
- Data loading and preprocessing utilities for tree-structured data.
- Tree and Node classes for representing constituency trees.
- An RNTN model for learning tree-structured representations.
- Training, evaluation, and embedding extraction routines.

Author: Cem Rifki Aydin
Date: 18.01.2020
"""

import ast
import os
import sys
import re
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Generator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Add project root to sys.path for imports
sys.path.append(os.path.join("..", ".."))

from ABSA_emb_gpu_final_newarch3 import FILES, REC_EMBEDDING_DIM, args as main_args

# Constants and configuration (hyperparameters)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = REC_EMBEDDING_DIM
ROOT_ONLY = True
BATCH_SIZE = 20
EPOCHS = main_args.constit_epochs  # 3
LR = 0.001

def flatten(l: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists."""
    return [item for sublist in l for item in sublist]

# -------------------- Data Structures --------------------

class Node:
    """A node in a binary constituency tree."""
    def __init__(self, label: int, word: Optional[str] = None):
        self.label = label
        self.word = word
        self.parent: Optional['Node'] = None
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None
        self.isLeaf = False

    def __str__(self):
        if self.isLeaf:
            return f'[{self.word}:{self.label}]'
        return f'({self.left} <- [{self.word}:{self.label}] -> {self.right})'

class Tree:
    """A binary constituency tree parsed from a string."""
    def __init__(self, tree_string: str, open_char: str = '(', close_char: str = ')'):
        self.open = open_char
        self.close = close_char
        tokens = self._tokenize(tree_string)
        self.root = self._parse(tokens)
        self.labels = self._get_labels(self.root)
        self.num_words = len(self.labels)

    def _tokenize(self, tree_string: str) -> List[str]:
        tokens = []
        for toks in tree_string.strip().split():
            tokens += list(toks)
        return tokens

    def _parse(self, tokens: List[str], parent: Optional[Node] = None) -> Node:
        if not tokens:
            raise ValueError("Empty token list for tree parsing.")
        assert tokens[0] == self.open and tokens[-1] == self.close, "Malformed tree"

        split = 2
        count_open = count_close = 0
        if tokens[split] == self.open:
            count_open += 1
            split += 1
        while count_open != count_close:
            if tokens[split] == self.open:
                count_open += 1
            if tokens[split] == self.close:
                count_close += 1
            split += 1

        node = Node(int(tokens[1]))
        node.parent = parent

        if count_open == 0:
            node.word = ''.join(tokens[2:-1]).lower()
            node.isLeaf = True
            return node

        node.left = self._parse(tokens[2:split], parent=node)
        node.right = self._parse(tokens[split:-1], parent=node)
        return node

    def get_words(self) -> List[str]:
        return [node.word for node in get_leaves(self.root)]

    def _get_labels(self, node: Optional[Node]) -> List[int]:
        if node is None:
            return []
        return self._get_labels(node.left) + self._get_labels(node.right) + [node.label]

# -------------------- Tree Utilities --------------------

def get_leaves(node: Optional[Node]) -> List[Node]:
    """Return all leaf nodes in a tree."""
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    return get_leaves(node.left) + get_leaves(node.right)

# -------------------- Data Loading --------------------

def load_trees(data: str = 'train') -> List[Tree]:
    """
    Load trees from file.
    If skip_marker is provided, lines containing it are skipped.
    """
    file_path = f'constituency/data/trees/{data}.txt'
    print(f"Loading {data} trees from {file_path}...")
    trees = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            trees.extend([Tree(tree) for tree in ast.literal_eval(line.strip())])  # Ensure the line is a valid tree string
    return trees

def load_eval_trees(data: str) -> List[List[Tree]]:
    """
    Load evaluation trees, grouped by aspect marker (###).
    """
    file_path = f'constituency/data/trees/{data}.txt'
    asp_groups = []
    print(f"Loading {data} trees from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        asp_group = []
        for line in f:
            if "###" in line:
                if asp_group:
                    asp_groups.append(asp_group)
                    asp_group = []
            else:
                asp_group.append(Tree(line))
        if asp_group:
            asp_groups.append(asp_group)
    return asp_groups

def get_batch(batch_size: int, data: List[Any]) -> Generator[List[Any], None, None]:
    """Yield batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# -------------------- Model --------------------

class RNTN(nn.Module):
    """
    Recursive Neural Tensor Network for tree-structured data.
    """
    def __init__(self, word2index: Dict[str, int], hidden_size: int, output_size: int):
        super().__init__()
        self.word2index = word2index
        self.embed = nn.Embedding(len(word2index), hidden_size)
        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size * 2, hidden_size * 2))
            for _ in range(hidden_size)
        ])
        self.W = nn.Parameter(torch.randn(hidden_size * 2, hidden_size))
        self.b = nn.Parameter(torch.zeros(1, hidden_size))
        self.W_out = nn.Linear(hidden_size, output_size)

    def init_weight(self):
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        for param in self.V.parameters():
            nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.W)
        self.b.data.fill_(0)

    def tree_propagation(self, node: Node, level: int = 0) -> OrderedDict:
        """
        Recursively compute representations for all nodes in the tree.
        Returns an OrderedDict mapping nodes to their embeddings.
        """
        recursive_tensor = OrderedDict()
        if node.isLeaf:
            idx = self.word2index.get(node.word, self.word2index['<UNK>'])
            tensor = torch.tensor([idx], dtype=torch.long, device=DEVICE)
            current = self.embed(tensor)
        else:
            left_tensors = self.tree_propagation(node.left, level + 1)
            right_tensors = self.tree_propagation(node.right, level + 1)
            recursive_tensor.update(left_tensors)
            recursive_tensor.update(right_tensors)
            left = recursive_tensor[node.left]
            right = recursive_tensor[node.right]
            concated = torch.cat([left, right], dim=1)
            xVx = []
            for v in self.V:
                xVx.append(torch.matmul(torch.matmul(concated, v), concated.transpose(0, 1)))
            xVx = torch.cat(xVx, 1)
            Wx = torch.matmul(concated, self.W)
            current = torch.tanh(xVx + Wx + self.b)
        recursive_tensor[node] = current
        return recursive_tensor

    def forward(self, trees: List[Tree], root_only: bool = False):
        """
        Forward pass for a batch of trees.
        Returns log probabilities and root tensors.
        """
        if not isinstance(trees, list):
            trees = [trees]
        propagated = []
        rec_tensors = []
        for tree in trees:
            recursive_tensor = self.tree_propagation(tree.root)
            if root_only:
                root_tensor = recursive_tensor[tree.root]
                propagated.append(root_tensor)
                rec_tensors.append(root_tensor)
            else:
                tensors = list(recursive_tensor.values())
                propagated.extend(tensors)
                rec_tensors.append(tensors[-1])
        propagated = torch.cat(propagated)
        return F.log_softmax(self.W_out(propagated), 1), rec_tensors

# -------------------- Training & Evaluation --------------------

def train(model: RNTN, train_data: List[Tree], lr: float, epochs: int):
    """
    Train the RNTN model.
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rescheduled = False

    for epoch in range(epochs):
        losses = []
        if not rescheduled and epoch == epochs // 2:
            lr *= 0.1
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            rescheduled = True

        for i, batch in enumerate(get_batch(BATCH_SIZE, train_data)):
            if ROOT_ONLY:
                labels = [tree.labels[-1] for tree in batch]
            else:
                labels = flatten([tree.labels for tree in batch])
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=DEVICE)

            model.zero_grad()
            preds, _ = model(batch, ROOT_ONLY)
            loss = loss_function(preds, labels_tensor)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'[{epoch+1}/{epochs}] mean_loss: {np.mean(losses):.2f}')
                torch.save(model, 'constituency/model/RecNN.pkl')
                losses = []
            if i > 200:
                return

def test(model: RNTN, test_data: List[Tree]):
    """
    Evaluate the model on test data.
    """
    accuracy = 0
    num_node = 0
    for test in test_data:
        model.zero_grad()
        preds, _ = model(test, ROOT_ONLY)
        labels = test.labels[-1:] if ROOT_ONLY else test.labels
        for pred, label in zip(preds.max(1)[1].data.tolist(), labels):
            num_node += 1
            if pred == label:
                accuracy += 1
    print(f'Test Accuracy: {accuracy / num_node * 100:.2f}%')

# -------------------- Embedding Extraction --------------------

def gold_aspects(data: List[Any]) -> Dict[int, List[str]]:
    """
    Extract gold aspect terms from data.
    """
    d = OrderedDict()
    row_cnter = cummul_cnter = 0
    while cummul_cnter < len(data):
        aspects = data[cummul_cnter][1]
        aspects = [
            re.sub("[ ]+", " ", re.sub(r"'([^s]|$)", r"\1", re.sub("(^')|('$)", "", 
                re.sub(r"(.){1}'s", r"\1 's", x.replace("\xc2\xa0", " ").replace("\\", " ").replace('[',"").replace("\"","").replace(']',"").strip().lower().replace("(", " -LRB- ").replace(")", " -RRB- "))))).strip()
            for x in aspects
        ]
        cummul_cnter += len(aspects)
        d[row_cnter] = aspects
        row_cnter += 1
    return d

def read_root_embs(model: RNTN, inp_file: str) -> List[List[Any]]:
    """
    Read root embeddings for each aspect group from file.
    """
    all_asp_root_embs = []
    asp_root_embs = []
    file_path = f'constituency/data/trees/{inp_file}.txt'
    with open(file_path, "r") as inp:
        lines = inp.readlines()
        lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
        for line in lines:
            model.zero_grad()
        
            tree_lst = ast.literal_eval(line.strip())
            for tree_str in tree_lst:
                _, root_tensor = model(Tree(tree_str), ROOT_ONLY)
                asp_root_embs.append(root_tensor[0].data.tolist())
            all_asp_root_embs.append(asp_root_embs)
            asp_root_embs = []
        if asp_root_embs:
            all_asp_root_embs.append(asp_root_embs)
    
    return all_asp_root_embs


def map_corr_asp(model: RNTN, inp_file: str, data: List[Any]) -> List[List[Any]]:
    """
    Map root embeddings to gold aspects, padding or truncating as needed.
    """
    mapped_embs = []
    all_asp_root_embs = read_root_embs(model, inp_file)
    gold_asps = gold_aspects(data)
    # print(gold_asps)
    if len(all_asp_root_embs) != len(gold_asps):
        raise Exception("Analysis for recursive embedding has worked incorrectly.")
    for i, _ in enumerate(gold_asps):
        asp_embs = all_asp_root_embs[i]
        golds = gold_asps[i]
        len_asp_embs = len(asp_embs)
        len_gold = len(golds)
        if len_asp_embs == len_gold:
            mapped_embs.append(asp_embs)
        elif len_asp_embs < len_gold:
            mapped_embs.append(asp_embs + [asp_embs[-1]] * (len_gold - len_asp_embs))
        else:
            mapped_embs.append(asp_embs[:len_gold])
    return mapped_embs

def write_root_embs(model: RNTN, inp_file: str, data: List[Any], out: str):
    """
    Write mapped root embeddings to output file.
    """
    list_of_tensors = map_corr_asp(model, inp_file, data)
    with open(out, mode='w', newline='') as file:

        for tensors in list_of_tensors:
            formatted_tensor_lst = []
            for tensor in tensors:
                formatted_tensor_lst.append(tensor[0])
            file.write(str(formatted_tensor_lst) + "\n")


# -------------------- Main Routine --------------------

def build_vocab(trees: List[Tree]) -> Dict[str, int]:
    """
    Build vocabulary from training trees.
    """
    vocab = set(flatten([t.get_words() for t in trees]))
    word2index = {'<UNK>': 0}
    for word in vocab:
        if word not in word2index:
            word2index[word] = len(word2index)
    return word2index

def main(all_data: List[Any]):
    """
    Main entry point for training and embedding extraction.
    """
    train_data = load_trees('train')
    word2index = build_vocab(train_data)
    model = RNTN(word2index, HIDDEN_SIZE, 5).to(DEVICE)
    model.init_weight()
    train(model, train_data, LR, EPOCHS)
    for ind, data in enumerate(all_data):
        inp_file = "train" if ind == 0 else "test"
        out_file = FILES[ind].replace(".csv", f"_dim_{HIDDEN_SIZE}_con_asp_embs.csv")
        write_root_embs(model, inp_file, data, out_file)

if __name__ == '__main__':
    pass