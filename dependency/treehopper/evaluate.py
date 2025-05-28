import os
import torch

import argparse

import train

from predict import load_best_model
from data.dataset import SSTDataset
import constants

import sys

sys.path.append(os.path.join("..", ".."))

from ABSA_emb_gpu_final_newarch3 import FILES, REC_EMBEDDING_DIM

import csv




tr_file = FILES[0] 
test_file = FILES[1]  

DOMAIN = tr_file[:tr_file.rindex("_") + 1]

FILES_ALL = [FILES[0], FILES[1]]

def prepare_emb_file(root_tensors, data, emb_file):
    """
    Prepare the embedding file for the root tensors."""
    asp_cnts = data  
    aspect_cnts = []
    for row in data:
        nb_aspects = len(row[1])
        aspect_cnts.append(nb_aspects)

    ind = 0
    asp_incr = aspect_cnts[0]
    list_of_tensors = []
    while True:
        end = ind + asp_incr
        tensors = root_tensors[ind:end]
        list_of_tensors.append(tensors)

        ind = end
        
        if end >= len(root_tensors):
            break
        asp_incr = aspect_cnts[end]

    # Save to file
    with open(emb_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for tensors in list_of_tensors:
            formatted_tensor = []
            for tensor in tensors:
                formatted_tensor.append(tensor[0])
            writer.writerow([formatted_tensor])
            



def eval(args, data, file):
    """Evaluate the model on the given data and file."""

    trainer_instance = load_best_model(args.model_path, args)
    
    if "train" in file:
        constants.TRAIN = True
    else:
        constants.TRAIN = False
    
    partition = file[:file.index(".csv")]
    partition = partition[partition.rindex("_") + 1:]

    
    test_dataset = SSTDataset(args.input, trainer_instance.model.vocab, num_classes=3)
    loss, accuracies, outputs, output_trees, root_tensors = trainer_instance.test(test_dataset)
    prepare_emb_file(root_tensors, data, DOMAIN + partition + "_dim_" + str(REC_EMBEDDING_DIM) + "_dep_asp_embs.csv")
    test_acc = torch.mean(accuracies)
    return test_acc, root_tensors


def main(all_data):
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analysis Trees - Evaluation')
    parser.add_argument('--model_path', help='Path to saved model', required=True)
    parser.add_argument('--input', help='Path to input directory', default="dependency/treehopper/train")
    args = train.set_arguments(parser, {})

    for ind, data in enumerate(all_data):
        print("Accuracy {}".format(eval(args, data, FILES[ind])[0]))

if __name__ == "__main__":
    main()
