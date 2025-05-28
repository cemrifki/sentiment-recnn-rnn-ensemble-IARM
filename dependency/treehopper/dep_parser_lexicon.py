#/usr/bin/py
# -*- coding: utf-8 -*-

import re
import csv
from copy import deepcopy
from collections import Counter


import spacy
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer()

nlp = spacy.load("en_core_web_sm")

SENTIMENT_LABELS = {"positive": 1, "neutral": 0, "negative": -1}


def sorted_asps(s):
    """
    Sorts the aspects in the input list of sets based on the index of the tokens.
    """
    
    sorted_asps = []
    for asp in s:
        l = list(asp)
        l.sort(key=lambda tok: int(tok[tok.rindex("-")+1:]))
        sorted_asps.append(l)
    return sorted_asps

def get_children_recurs(tok, lev):
    """
    Recursively retrieves all children of a token in a dependency tree.
    """

    all_children = set([])
    if lev == 0:
        all_children.add(tok.text + "-" + str(tok.i))
    lev += 1
    for kid in tok.children:
        tag = kid.tag_.lower()
        dep = kid.dep_.lower()
        if (dep == "conj" or dep == "ccomp") and (tag == "vbd" or tag == "vbz" or tag == "vbg"): continue
        all_children.add(kid.text + "-" + str(kid.i))
        all_children |= get_children_recurs(kid, lev)
    return all_children

def get_all_asps(doc):
    """
    Retrieves all aspects from a document by extracting the children of each token.
    """
    res = []
    for token in doc:
        kids = get_children_recurs(token, 0)
        res.append(kids)
    return res

def map_rev_w_asps(rev, gold_asps):
    """
    Maps the review to its aspects by extracting subtrees from the dependency tree.
    """
    doc = nlp(rev)
    asps = get_all_asps(doc)
    sorted_aspects = sorted_asps(asps) #yanlis
    gold_aspects_w_ind = gold_asps_w_ind(doc, gold_asps)    
    sub_revs = []
    for gold_aspect_w_ind in gold_aspects_w_ind:
        matched = get_match_w_max_len(sorted_aspects, gold_aspect_w_ind)
        matched = [word[:word.rindex("-")] for word in matched]
        sub_revs.append(matched)
    return sub_revs

def write_subreviews_per_aspect(file):
    """
    Reads the data, extracts dependency features,
    and writes sub-reviews for each aspect to a text file.
    """
    outfile_name = file.replace(".csv", "_subreviews.txt")
    revs_dep_feats = get_dep_features(file)
    with open(outfile_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for rev_dep_feats in revs_dep_feats:

            aspect_subtrees = rev_dep_feats[0]

            writer.writerow([" ".join(aspect_subtree).strip() for aspect_subtree in aspect_subtrees])

def generate_asps_from_rev_deps(feats):
    """
    Generates aspect subtrees from reviews and their corresponding
    """
    res = []
    for ind, line in enumerate(feats):
        rev_all = []
        rev = line[0]
        rev_gold_asps = line[1]
        rev_sentiments = line[2]

        aspect_subtrees = map_rev_w_asps(rev, rev_gold_asps)
        rev_all.append(aspect_subtrees)
        rev_all.append(rev_gold_asps)
        rev_all.append(rev_sentiments)
        res.append(rev_all)
    return res

def get_dep_features(file):
    """
    Extract dependency features and returns the modified
    components with respect to a review file.
    """       
    data = csv_reader(file)
    dep_features = generate_asps_from_rev_deps(data)
    return dep_features
    
def get_match_w_max_len(sorted_asps, gold_asp_w_ind):
    """
    Finds the aspect subtree with the maximum length that matches the gold aspect.
    """
    
    max_len = -1
    asp_tree_max = []
    for sorted_asp in sorted_asps:
        l = find_sub_list(gold_asp_w_ind, sorted_asp)
        len_ = len(sorted_asp)
        if l and len_ > max_len:
            max_len = len_
            asp_tree_max = sorted_asp
    if max_len > 0: 
        return asp_tree_max
    return sorted_asps[0]


def gold_asps_w_ind(doc, gold_asps):
    """
    Maps the gold aspects to their indices in the document.
    """
    rev = [tok.text for tok in doc]
    used_asps = Counter()
    gold_asps_w_ind = []
    
    rev_for_char_unmatch = deepcopy(rev)
    
    for gold_asp in gold_asps:
        gold_asp_l = gold_asp.split()
        matches = find_sub_list(gold_asp_l, rev)
        failed1, failed2, failed3 = [False] * 3
        if not matches:
            failed1 = True
            gold_asp_l = [asp_tok.text for asp_tok in nlp(gold_asp)]
            
            matches = find_sub_list(gold_asp_l, rev)
            
            if not matches:
                failed2 = True
                gold_asp_l = gold_asp_l[:-1]
                matches = find_sub_list(gold_asp_l, rev)
                if not matches:
                    failed3 = True
                    matches = [(0, min(len(rev), len(rev_for_char_unmatch)))]
        
        ind_tuple = matches[used_asps[gold_asp]]
        st_ind, end_ind = ind_tuple[0], ind_tuple[-1]
        gold_asp_w_ind = [rev[ind] + "-" + str(ind) for ind in range(st_ind, end_ind)]
        gold_asps_w_ind.append(gold_asp_w_ind)
        if len(matches) > 1:
            used_asps[gold_asp] += 1
        rev = rev_for_char_unmatch
    return gold_asps_w_ind

def find_sub_list(sl, l):
    """
    Finds all occurrences of a sublist in a list and returns their start and end indices.
    """
    results = []
    sll = len(sl)
    if not sll: return results
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll] == sl:
            results.append((ind, ind+sll))

    return results


def csv_reader(data):
    """
    Read the input csv file and generates reviews, sentiments, and aspects.
    """ 

    refined_revs = []
    pat = re.compile("[\'\" ]+,")
    cnt = 1
    for row in data: 
        refined_rev = []
        cnt += 1
        sent = row[0].lower()
        sent = re.sub("'(\\s+|$)|(\\s+|$)'", " ", sent)
        sent = re.sub(r"(\w+)'([^s]+)", r"\1\2", sent)
        sent = re.sub("(\"+\\s+)|(\\s+\"+)", " ", sent)
        sent = re.sub(r"([\.]+)", r"\1 ", sent)
        sent = " ".join([tok.text for tok in nlp(sent)])
        sent = re.sub("[ ]+", " ", sent)
        row_upd = row[2].replace(u"\\xc2\\xa0", u" ")
        row_upd = re.sub(r"([0-9]+\.[0-9]+)", r"\1 ", row_upd)
        aspects = [re.sub("[ ]+", " ", re.sub(r"'([^s]|$)", r"\1", re.sub("(^')|('$)", "", re.sub(r"(.){1}'s", r"\1 's", x.replace("\xc2\xa0", " ").replace("\\", " ").replace('[',"").replace("\"","").replace(']',"").strip().lower())))).strip() for x in pat.split(row_upd)]
        
        tmp_aspects = []
        for asp in aspects:
            tmp_aspects.append(" ".join([tok.text for tok in nlp(asp)]))
        aspects = tmp_aspects
                
        
        sentiments = row[3]  # [x.strip().replace("'","").replace('[',"").replace("\"","").replace(']',"").lower() for x in row[3].split(",")]

        sentiments = [SENTIMENT_LABELS[sentiment] for sentiment in sentiments]
        
        refined_rev.append(sent)
        refined_rev.append(aspects)
        refined_rev.append(sentiments)
        
        refined_revs.append(refined_rev)
        
    return refined_revs


if __name__ == "__main__":
    write_subreviews_per_aspect("2014_Laptop_train.csv")
    write_subreviews_per_aspect("2014_Laptop_test.csv")
