# -*- coding: utf-8 -*-
#!/usr/bin/py
"""
Constituency Parsing and Aspect Group Extraction Module

This module provides functions for extracting aspect groups from constituency parse trees,
handling multi-word aspects, normalizing parses, and integrating with Stanford CoreNLP for
constituency parsing. It is designed for aspect-based sentiment analysis tasks.

Author: Cem Rifki Aydin
Date: 15.01.2020
"""
import ast
import glob
import os
import re
import regex as reg
import copy
import subprocess
import sys
from collections import defaultdict, OrderedDict, Counter

import spacy
import pandas as pd

import RecNN

sys.path.append(os.path.join("..", ".."))
from ABSA_emb_gpu_final_newarch3 import FILES

# -------------------- Configuration --------------------

MODEL_NAME = "en_core_web_sm"
CONJUNCTIONS = ["and", "or", "but", "however", "also", ".", ",", ";", ":", "?", "!"]
SENTIMENT_SCORES = ["0", "1", "2", "3", "4"]
POLARITY = ["0", "1", "3", "4"]

# -------------------- Utility Functions --------------------

def load_spacy_model(model_name: str):
    """
    Load a spaCy language model, installing it if necessary.
    """
    from spacy.util import is_package
    try:
        if not is_package(model_name):
            print(f"Model '{model_name}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        nlp = spacy.load(model_name)
        print(f"Model '{model_name}' loaded successfully.")
        return nlp
    except Exception as e:
        print(f"Error loading or installing model: {e}")
        raise

nlp = load_spacy_model(MODEL_NAME)

def words_and_count(s: str):
    """
    Extract words from a string and return the list and its length.
    """
    regex = r'\b[a-zA-Z\'é\.\!\?\']+\b'
    l = re.findall(regex, s, re.UNICODE)
    return l, len(l)

def reduce_white_space_from_list(l):
    """
    Remove extra whitespace from each string in a list.
    """
    return [re.sub("[\\s]+", " ", asp.strip()) for asp in l]

def normalise_parse(s: str):
    """
    Normalize a parse string by removing parse numbers and extra characters.
    """
    s = re.sub(r"\([0-5]+", "", s)
    s = re.sub(r"[\(\)\"\`]+", "", s)
    s = re.sub(r"(.){1}'s", r"\1 's", s)
    s = re.sub(r"([']+)([^s])", r"\2", s)
    return re.sub(r"[ ]+", " ", s)

def append_cum(l):
    """
    Compute cumulative sum of a list.
    """
    res = copy.deepcopy(l)
    for i in range(1, len(l)):
        res[i] += res[i - 1]
    return res

def find_range(i, l):
    """
    Find the index in list l where i would fit.
    """
    for ind, it in enumerate(l):
        if i < it:
            return ind

def pat():
    """
    Return a compiled regex pattern for splitting.
    """
    return re.compile("[\'\" ]+,")

# -------------------- Aspect Extraction Core --------------------

def include_all_aspect_terms(s, multi_word_asp, rec_cnt):
    """
    Ensure all aspect terms are included in the parse subtree.
    """
    regex = "".join([w + ".*?" for w in multi_word_asp])
    regex = regex[:regex.rfind(".*?")]
    matches = reg.finditer(regex, s, overlapped=True)
    results = [match.group(0) for match in matches]
    if not results:
        return s
    min_reg_ind = min(range(len(results)), key=lambda i: len(results[i]))
    regex = results[min_reg_ind]
    st = s.index(regex)
    end = st + len(regex)
    multiwords_cands = []
    for word in multi_word_asp:
        for m in re.finditer(r"\([0-4] " + word + "\)", s):
            word_st = m.start() + 3
            word_end = m.end() - 1
            if word_st >= st and word_end < end:
                word_st = word_st - 3
                word_end = word_st + len(word) + 3 + 1
                scope = extr_asp_tr_bottom_up_helper(s, word_st, word_end, False, False, True, multi_word_asp)
                multiwords_cands.append(scope)
    if not multiwords_cands:
        return s
    min_ind = min(range(len(multiwords_cands)), key=lambda i: len(multiwords_cands[i]))
    return multiwords_cands[min_ind]

def findnth(s, tok, n):
    """
    Find the nth occurrence of token 'tok' in string 's'.
    """
    tmp_tok = " " + tok + ")"
    parts = s.split(tmp_tok, n+1)
    if len(parts) <= n+1:
        return -1
    return len(s)-len(parts[-1])-len(tmp_tok)

def findnth_single_word(s, tok, n):
    """
    Find the nth occurrence of a single word token in string 's'.
    """
    parts = s.split(tok, n+1)
    if len(parts) <= n+1:
        return -1
    return len(s)-len(parts[-1])-len(tok)

def extr_asp_tr_bottom_up(s, gold, rec_cnt):
    """
    Extract aspect subtree for a given aspect term.
    """
    gold = gold.strip()
    asp_terms = gold.split()
    if len(asp_terms) > 1:
        return include_all_aspect_terms(s, asp_terms, rec_cnt)
    if gold in rec_cnt:
        st = findnth(s, gold, rec_cnt[gold])
    else:
        word = " " + gold + ")"
        if word not in s:
            return s
        st = s.index(word)
    st = st - 2
    end = st + len(gold) + 4
    contr = ("0" in s or "1" in s) and ("3" in s or "4" in s)
    neg = " not)" in s or " n't)" in s
    return extr_asp_tr_bottom_up_helper(s, st, end, contr, neg, False, [])

def extr_asps_tr_bottom_up(s, golds):
    """
    Extract aspect subtrees for all gold aspect terms in a sentence.
    """
    asp_rec_trees_flatten = Counter([word for gold in golds for word in gold.split()])
    asp_rec_trees_dict = {asp: range(cnt) for asp, cnt in asp_rec_trees_flatten.items() if cnt > 1}
    asp_rec_trees = []
    for gold in golds:
        gold = gold.strip()
        asp_terms = gold.split()
        recs = {}
        for term in asp_terms:
            if term in asp_rec_trees_dict:
                rec = asp_rec_trees_dict[term][0]
                recs[term] = rec
        asp_rec_tree = extr_asp_tr_bottom_up(s, gold, recs)
        asp_rec_trees.append(asp_rec_tree)
        if asp_terms:
            for term in asp_terms:
                if term in asp_rec_trees_dict:
                    asp_rec_trees_dict[term] = asp_rec_trees_dict[term][1:]
    return asp_rec_trees

def extr_asp_tr_bottom_up_helper(s, st, end, contr, neg, is_multi_word, multiwords):
    """
    Helper for extracting aspect subtree, handling multi-word and negation.
    """
    res = ""
    left_par_cnt = right_par_cnt = 0
    if end == len(s):
        return s
    st -= 1
    if s[end] == ")":
        while st >= 0:
            if not s[st]:
                st -= 1
                continue
            if s[st] == ")":
                left_par_cnt += 1
            elif s[st] == "(":
                left_par_cnt -= 1
            if left_par_cnt == -1:
                end += 1
                res = s[st:end]
                break
            st -= 1
    else:
        while end < len(s):
            if not s[end]:
                end += 1
                continue
            if s[end] == "(":
                right_par_cnt += 1
            elif s[end] == ")":
                right_par_cnt -= 1
            if right_par_cnt == -1:
                st -= 2
                end += 1
                res = s[st:end]
                break
            end += 1
    has_rev_dep_features = has_dep_features(res)
    if is_multi_word:
        incl_all = all([w in res for w in multiwords])
        if incl_all:
            if not has_rev_dep_features:
                return extr_asp_tr_bottom_up_helper(s, st, end, contr, neg, is_multi_word, multiwords)
            return res
        else:
            return extr_asp_tr_bottom_up_helper(s, st, end, contr, neg, is_multi_word, multiwords)
    else:
        if not has_rev_dep_features:
            return extr_asp_tr_bottom_up_helper(s, st, end, contr, neg, is_multi_word, multiwords)
        else:
            if neg_scope(s, res):
                return extr_asp_tr_bottom_up_helper(s, st, end, contr, neg, is_multi_word, multiwords)
    return res

def has_dep_features(s):
    """
    Check if a string contains both a verb and a subject (using spaCy).
    """
    s = re.sub(r"\([0-4]+ ", "", s)
    s = s.replace("(", "").replace(")", "")
    str_stripped_of_pars = re.sub("[ ]+", " ", s).strip()
    doc = nlp(str_stripped_of_pars)
    has_verb, has_subj = False, False
    for token in doc:
        if token.pos_.lower() == "verb":
            has_verb = True
        if token.dep_.lower() == "nsubj":
            has_subj = True
    return has_verb and has_subj

def neg_scope(rev, asp):
    """
    Determine if a negation is in the scope of the aspect.
    """
    neg = [" n't)", " not)"]
    st = rev.index(asp)
    prefix = rev[:st]
    last_neg_ind = -1
    fnd = False
    for n in neg:
        i = prefix.rfind(n)
        if i > last_neg_ind:
            last_neg_ind = i
        if n in rev:
            fnd = True
    if not fnd:
        return False
    l, len_ = words_and_count(prefix[last_neg_ind:])
    return len_ <= 2

# -------------------- Aspect Group Extraction --------------------

def extr_rec_sent_asp_groups(s_exp, golds):
    """
    Extract aspect groups from a parse tree string.
    """
    asp_groups = []
    cnt_last_paren = 0
    cur_subtree = ""
    last_asp_gr = ""
    for i in range(len(s_exp) - 1, -1, -1):
        ch = s_exp[i]
        cur_subtr_words, cur_subtr_cnt = words_and_count(cur_subtree)
        if ch == "(":
            if cur_subtr_words in CONJUNCTIONS:
                cur_subtree = ""
                cnt_last_paren = 0
                last_asp_gr = ""
                continue
            cnt_last_paren -= 1
            cur_subtree = "(" + cur_subtree
            last_asp_gr = "(" + last_asp_gr
            if cnt_last_paren == 1:
                asp_words, asp_word_cnt = words_and_count(last_asp_gr)
                if cur_subtr_cnt == 0 or asp_word_cnt == 0:
                    cur_subtree, last_asp_gr = "", ""
                    cnt_last_paren = 0
                    continue
                if asp_word_cnt < 2:
                    if asp_words[0] not in CONJUNCTIONS:
                        asp_groups.insert(0, last_asp_gr)
                elif asp_word_cnt == 2:
                    cur_subtree = ""
                    cnt_last_paren = 0
                    asp_groups.insert(0, last_asp_gr.strip()[:-1])
                    last_asp_gr = ""
                else:
                    last_asp_gr = ""
                    asp_mod = cur_subtree + " " + last_asp_gr
                    asp_groups.insert(0, asp_mod.strip()[:-1])
                last_asp_gr = ""
                cur_subtree = ""
                cnt_last_paren = 0
                init_substr = str(s_exp[:i])
                init_substr = re.sub(r"[ ]+", "", init_substr)
                init_substr = re.sub(r"\)\(", "--", init_substr)
                init_substr = re.sub(r"[ 012345\\(\\)]+", "", init_substr)
                words, cnt_words = words_and_count(init_substr)
                if cnt_words <= 3:
                    remain_first_asp_gr = asp_groups[0]
                    if words_and_count(remain_first_asp_gr)[1] > 2:
                        asp_groups.insert(0, str(s_exp[:i]))
                        break
                    remain_first_asp_gr = handle_first_asp_gr_initials(s_exp[:i]) + " " + remain_first_asp_gr
                    subs_asp = handle_subseq_asp(s_exp, remain_first_asp_gr)
                    remain_first_asp_gr = remain_first_asp_gr.strip() + ")"
                    if len(subs_asp) > 0:
                        remain_first_asp_gr = remain_first_asp_gr + " " + subs_asp
                    asp_groups[0] = remain_first_asp_gr
                    asp_groups = reduce_white_space_from_list(asp_groups)
                    asp_groups = check_form(asp_groups, s_exp)
                    matched = match_gold_asp_w_pred(golds, asp_groups)
                    return matched
        elif ch == ")":
            cur_subtree = ch + cur_subtree
            last_asp_gr = ch + last_asp_gr
            cnt_last_paren += 1
        else:
            cur_subtree = ch + cur_subtree
            last_asp_gr = ch + last_asp_gr
    asp_groups = check_form(asp_groups, s_exp)
    asp_groups = reduce_white_space_from_list(asp_groups)
    matched = match_gold_asp_w_pred(golds, asp_groups)
    return matched

def match_gold_asp_w_pred(gold, pred):
    """
    Match gold aspect terms with predicted aspect groups.
    """
    matched_asps = []
    d = defaultdict(int)
    for ind, asp in enumerate(gold):
        match_inds = [cnt_asp_in_parse(asp, pred_asp) if match_skip_par_nos(asp, pred_asp) else 0 for i, pred_asp in enumerate(pred)]
        match_inds = append_cum(match_inds)
        pred_asp_ind = find_range(d[asp], match_inds)
        matched_asps.append(pred[pred_asp_ind])
        d[asp] += 1
    return matched_asps

def match_skip_par_nos(gold, pred):
    """
    Check if all tokens in gold aspect are present in predicted aspect group.
    """
    spl = gold.split()
    if len(spl) > 1:
        return all([token in pred for token in spl])
    else:
        return spl[0] in pred

def cnt_asp_in_parse(asp, parse):
    """
    Count occurrences of aspect in normalized parse.
    """
    return normalise_parse(parse).count(asp)

def check_form(asp_groups, s_exp):
    """
    Ensure all aspect groups are in correct parse form.
    """
    if not check_all_correct_forms(asp_groups):
        asp_groups = [s_exp]
        asp_groups = reduce_white_space_from_list(asp_groups)
    if not check_all_parse_form(asp_groups):
        asp_groups = [s_exp]
        asp_groups = reduce_white_space_from_list(asp_groups)
    return asp_groups

def check_all_correct_forms(asp_grs):
    """
    Check if all aspect groups have balanced parentheses.
    """
    return all([check_correct_form(asp_gr) for asp_gr in asp_grs])

def check_correct_form(asp):
    """
    Check if a string has balanced parentheses.
    """
    cnt_paren = 0
    for ch in asp:
        if ch == "(":
            cnt_paren += 1
        elif ch == ")":
            cnt_paren -= 1
        if cnt_paren < 0:
            return False
    return cnt_paren == 0

def check_all_parse_form(asp_grs):
    """
    Check if all aspect groups can be parsed as trees.
    """
    return all([check_parse_form(asp_gr) for asp_gr in asp_grs])

def check_parse_form(asp):
    """
    Check if a string can be parsed as a RecNN tree.
    """
    try:
        _ = RecNN.Tree(asp)
    except AssertionError:
        return False
    return True

def check_correct_forms_in_file(f):
    """
    Check all lines in a file for correct parse form.
    """
    with open(f, "r") as file:
        cnt = 0
        for line in file:
            # if "###" in line:
            #     continue
            if not check_correct_form(ast.literal_eval(line)[0]):
                cnt += 1

def handle_subseq_asp(s, detected_asp):
    """
    Handle subsequent aspect groups in a parse string.
    """
    if detected_asp not in s:
        return ""
    st_ind = s.index(detected_asp)
    later_asp_st_ind = st_ind + len(detected_asp) + 1
    cnt_r = cnt_l = 0
    res = ""
    for i in range(later_asp_st_ind, len(s)):
        if s[i] == ")":
            cnt_r += 1
        elif s[i] == "(":
            cnt_l += 1
        res += s[i]
        if cnt_l >= 2:
            return ""
        if cnt_r == 2:
            break
    return res.strip()

def handle_first_asp_gr_initials(s):
    """
    Handle the initial part of the first aspect group in a parse string.
    """
    s = s.strip()
    cnt_paren = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ")":
            cnt_paren += 1
        else:
            break
    cnt_paren += 2
    words, count = words_and_count(s)
    word_start_ind = s.index(words[0])
    asp_word_start_str = s[word_start_ind:]
    pre_asp = s[:word_start_ind]
    for i in range(len(pre_asp) - 1, -1, -1):
        cur_ch = pre_asp[i]
        if cur_ch == "(":
            cnt_paren -= 1
        asp_word_start_str = cur_ch + asp_word_start_str
        if cnt_paren == 0:
            break
    return asp_word_start_str

# -------------------- File I/O and Data Preparation --------------------

def get_gold_aspects(data):
    """
    Extract gold aspect terms from data.
    """
    row_cnter = cummul_cnter = 0
    all_aspects = []
    while cummul_cnter < len(data):
        aspects = data[cummul_cnter][1]
        cummul_cnter += len(aspects)
        aspects = [re.sub("[ ]+", " ", re.sub(r"'([^s]|$)", r"\1", re.sub("(^')|('$)", "", re.sub(r"(.){1}'s", r"\1 's", x.replace("\xc2\xa0", " ").replace("\\", " ").replace('[',"").replace("\"","").replace(']',"").strip().lower().replace("(", " -LRB- ").replace(")", " -RRB- "))))).strip()
                   for x in aspects]
        all_aspects.append(aspects)
        row_cnter += 1
    return all_aspects

def extract_revs_only(data, file):
    """
    Extract only the review texts from data and write to a file.
    """
    revs = []
    row_cnter = cummul_cnter = 0
    while cummul_cnter < len(data):
        review = data[cummul_cnter][0]
        cummul_cnter += len(data[cummul_cnter][1])
        revs.append(review)
    revs_only_file_name = file.replace(".csv", "_revs.txt")
    with open(revs_only_file_name, "w", encoding="utf-8") as f:
        for rev in revs:
            f.write(rev.strip('"') + "\n")
    return revs_only_file_name

def extract_asp_grs_from_file(file, asps):
    """
    Extract aspect groups from a file of parse trees.
    """
    d = OrderedDict()
    cnt = 0
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            gold_asps = asps[cnt]
            asp_grs = extr_asps_tr_bottom_up(line, gold_asps)
            d[cnt] = asp_grs
            cnt += 1
    return d

def write_asp_groups_2_file(data, in_file, out_file):
    """
    Write extracted aspect groups to a file.
    """
    gold_asps = get_gold_aspects(data)
    asp_grs = extract_asp_grs_from_file(in_file, gold_asps)

    with open(out_file, "w", newline='') as file:

        for id, rev_asps in asp_grs.items():
            formatted_tensor = []
            for rev_asp in rev_asps:
                rev_asp_stripped = rev_asp.strip('"\'').strip()
                formatted_tensor.append(rev_asp_stripped)
                
            file.write(str(formatted_tensor) + "\n")

# -------------------- Main Pipeline --------------------

def run_corenlp_parser(input_file, output_file, corenlp_dir):
    """
    Run Stanford CoreNLP constituency parser on input_file and write output to output_file.
    """
    jar_files = glob.glob(os.path.join(corenlp_dir, "*.jar"))
    classpath = ":".join(jar_files)
    print(f"Constituency parser files (pennTree's) for {input_file} are being generated.")
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        subprocess.run([
            "java",
            "-cp", classpath,
            "-mx5g",
            "edu.stanford.nlp.sentiment.SentimentPipeline",
            "-stdin",
            "-output", "pennTrees"
        ], stdin=infile, stdout=outfile, stderr=subprocess.PIPE)

def main(training_data=None, test_data=None):
    """
    Main function to process training and test data, extract aspect groups, and write results.
    """
    for ind, data in enumerate([training_data, test_data]):
        file_name = FILES[ind]
        domain = file_name[file_name.rfind("_") + 1:file_name.index(".csv")]
        revs_only_file_name = extract_revs_only(data, file_name)
        penn_file_name = file_name.replace(".csv", "_pennTrees.txt")
        corenlp_dir = "constituency/stanford-corenlp-4.5.8"
        run_corenlp_parser(revs_only_file_name, penn_file_name, corenlp_dir)
        output_tree_file = f"constituency/data/trees/{domain}.txt"
        write_asp_groups_2_file(data, penn_file_name, output_tree_file)
        # check_correct_forms_in_file(output_tree_file)

if __name__ == "__main__":
    main()
