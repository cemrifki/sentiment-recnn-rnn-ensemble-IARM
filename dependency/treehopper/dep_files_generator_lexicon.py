#!/usr/bin/py
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:10:53 2019

@author: Cem Rifki Aydin

"""
import os
import copy


import json


import spacy
from spacy.util import is_package

from spacy.cli import download

MODEL_NAME = "en_core_web_sm"

# Check if the model is already installed
if not is_package(MODEL_NAME):
    print(f"Model '{MODEL_NAME}' not found. Downloading...")
    download(MODEL_NAME)

nlp = spacy.load("en_core_web_sm")

import constants
import dep_parser_lexicon
import os


import sys
sys.path.append(os.path.join("..", ".."))

from ABSA_emb_gpu_final_newarch3 import FILES


def get_domain(s):
    """
    Extracts the domain (Laptop or Restaurants) from the filename.
    The domain is assumed to be the part of the filename between the last underscore and the last dot.
    """
    underscore_ind = s.rindex("_") + 1
    ext_ind = s.rindex(".")
    return s[underscore_ind:ext_ind]

def get_lex_scores(file):
    """
    Reads a JSON file containing sentiment lexicon scores and returns a dictionary with the scores.
    """

    with open(file, "r") as json_file:
        data = json.load(json_file)
    data_upd = copy.deepcopy(data)
    for k, v in data.items():
        if k[0] == "#":
            data_upd[k[1:]] = v
    return data_upd


def generate_parents(data, in_file, out_fold):
    """
    Generates and writes parent indices for each word in the input data using a dependency parser.
    """
    domain = get_domain(in_file)
    is_train = "train" in domain.lower()
    out_file = ("rev" if is_train else "evaltest") + "_parents.txt"
    out_file = os.path.join(out_fold, out_file)
    revs = [datum[0] for datum in data] 
    all_subrevs_parents = []
    with open(out_file, "w") as o:
        for rev in revs:
            subrevs_parents = []
            for subrev in rev:
                subrev = " ".join(subrev)
                doc = nlp(subrev)
                subrev_parents = [0 if tok.dep_.lower() == "root" else
                                  tok.head.i for tok in doc]
                subrevs_parents.append(subrev_parents)
                subrev_parents_str = [str(tok) for tok in subrev_parents]
                o.write(" ".join(subrev_parents_str) + "\n")
            all_subrevs_parents.append(subrevs_parents)
    return all_subrevs_parents

def get_root_indices(data, in_file, out_fold):
    """
    Generates root indices for each sub-review in the input data.
    """
    parents = generate_parents(data, in_file, out_fold)
    all_root_inds = []
    for rev in parents:
        rev_root_inds = []
        for subrev in rev:
            root_ind = subrev.index(0)
            rev_root_inds.append([root_ind])
        all_root_inds.append(rev_root_inds)
    return all_root_inds
def generate_labels(data, in_file, out_fold):
    """
    Generates and writes word-level polarity labels for each word in the input data using a sentiment lexicon.

    This function processes a dataset of reviews (or similar text data), assigns a polarity score to each word
    based on a predefined lexicon, and writes the resulting labels to a file. The output file is named according
    to whether the input is training or evaluation data. The function also returns all generated labels as a nested list.

    Args:
        data (list): A list of data samples, where each sample is expected to be a tuple or list. The first element
            (datum[0]) should be a list of sub-reviews (sentences or text segments), each of which is a list of words.
            The third element (datum[2]) is assumed to be the gold labels for the review (not directly used in labeling).
        in_file (str): Path to the input file, used to determine the domain and whether the data is for training or evaluation.
        out_fold (str): Path to the output folder where the label file will be saved.

    Returns:
        list: A nested list of labels. The outer list corresponds to reviews, each containing a list of sub-reviews,
            each of which contains a list of string polarity scores (one per word).

    Side Effects:
        Writes a file to `out_fold` named either "rev_labels.txt" (for training data) or "evaltest_labels.txt"
        (for evaluation/test data), where each line contains space-separated polarity scores for the words in a sub-review.

    Notes:
        - The function relies on several helper functions and constants:
            - `get_domain(in_file)`: Determines the domain/type of the input file.
            - `get_root_indices(data, in_file, out_fold)`: Retrieves root indices for each review (usage context-specific).
            - `get_lex_scores(constants.ENG_SENT_LEXICON_FILE)`: Loads a dictionary mapping words to polarity scores.
        - Words not found in the lexicon are assigned a default score of 0.
    """

    domain = get_domain(in_file)
    is_train = "train" in domain.lower()
    out_file = ("rev" if is_train else "evaltest") + "_labels.txt"
    out_file = os.path.join(out_fold, out_file)

    root_inds = get_root_indices(data, in_file, out_fold)

    lex_scores = get_lex_scores(constants.ENG_SENT_LEXICON_FILE)
    revs = [datum[0] for datum in data]  

    revs_gold_lbls = [datum[2] for datum in data]  

    all_labels = []
    with open(out_file, "w") as o:
        for rev_ind, rev in enumerate(revs):
            subrevs_labels = []
            rev_root_inds = root_inds[rev_ind]
            rev_gold_lbls = revs_gold_lbls[rev_ind]
            for subrev_ind, subrev in enumerate(rev):
                subrev_labels = []
                for word_ind, word in enumerate(subrev):
                    word_pol_score = lex_scores.get(word, 0)
                    subrev_labels.append(str(word_pol_score))
                subrevs_labels.append(subrev_labels)
                o.write(" ".join(subrev_labels) + "\n")
            all_labels.append(subrevs_labels)

    return all_labels

def generate_sents(data, in_file, out_fold):
    """ 
    Generates and writes sentences from the input data to a file.
    """
    revs = [datum[0] for datum in data]  
    domain = get_domain(in_file)
    is_train = "train" in domain.lower()
    out_file = ("rev" if is_train else "evaltest") + "_sentence.txt"
    out_file = os.path.join(out_fold, out_file)
    with open(out_file, "w") as o:
        for rev in revs:
            for subrev in rev:

                subrev_str = " ".join([str(word) for word in subrev])
                o.write(subrev_str + "\n")
                
    return revs

def generate_rels(data, in_file, out_fold):
    """
    Generates and writes dependency relations for each word in the input data using a dependency parser.
    """
    revs = [datum[0] for datum in data]  
    all_subrevs_deps = []
    domain = get_domain(in_file)
    is_train = "train" in domain.lower()
    out_file = ("rev" if is_train else "evaltest") + "_rels.txt"
    out_file = os.path.join(out_fold, out_file)

    with open(out_file, "w") as o:
        for rev in revs:
            subrevs_deps = []
            for subrev in rev:
                subrev = " ".join(subrev)
                doc = nlp(subrev)

                subrev_deps = [tok.dep_ for tok in doc]
                subrevs_deps.append(subrev_deps)
                o.write(" ".join(subrev_deps) + "\n")
            all_subrevs_deps.append(subrevs_deps)
    return all_subrevs_deps

all_feat_generators = [generate_labels, generate_parents, generate_rels, generate_sents]

def write_feats_2_files(data, in_file_name, out_fold):
    """
    Writes various features extracted from the input data to files in the specified output folder.
    """
    out_fold = os.path.join("dependency", "treehopper", out_fold)
    if not os.path.exists(out_fold): os.makedirs(out_fold)
    data_dep_feats = dep_parser_lexicon.get_dep_features(data)
    for feat_generator in all_feat_generators:
        feat_generator(data_dep_feats, in_file_name, out_fold)


def main(all_data):
    
    for ind, data in enumerate(all_data):
        print(f"Dependency files are being generated for {FILES[ind]}.")
        write_feats_2_files(data, FILES[ind], "training-treebank")  # "train" if ind == 0 else "test")

if __name__ == "__main__":
    main()
