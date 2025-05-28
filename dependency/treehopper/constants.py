import os

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


ENG_SENT_LEXICON_FILE = os.path.join("dependency", "treehopper", "twitter_sentiments.json")

TRAIN = True