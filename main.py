import sys
sys.path.append('.')

import argparse
import torch
import json
import os

from training import train, evaluate
from models.seq2seq import Seq2Seq
from torch.utils import data

import utils.load_raw_data as dl
from utils.data_generator import MyData, LanguageIndex
import utils.preprocess as dp

from sklearn.model_selection import train_test_split

def run():

    ## Load Config from JSON file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    config_path = os.path.join(dir_path,"experiments", FLAGS.config)

    if not os.path.exists(config_path):
        raise FileNotFoundError

    with open(config_path, "r") as f:
        config = json.load(f)

    config["gpu"] = torch.cuda.is_available()
    
    ## Load Data
    df = dl.load_raw_text_file(FLAGS.data_path,num_examples=30000)

    # index language for Input and Output
    inp_index = LanguageIndex(phrases=df["es"].values)
    targ_index = LanguageIndex(df["eng"].values)
    vocab_inp_size = len(inp_index.word2idx)
    vocab_tar_size = len(targ_index.word2idx)

    # Convert Sentences into tokenized tensors
    input_tensor,target_tensor = dl.convert_tensor(df,inp_index,targ_index)
    # Split to training and test set
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
    train_dataset = MyData(input_tensor_train, target_tensor_train)
    val_dataset = MyData(input_tensor_val, target_tensor_val)

    # Conver to DataLoader Object
    train_dataset = DataLoader(train_dataset,
                     batch_size=BATCH_SIZE, 
                     drop_last=True,
                     shuffle=True)

    eval_dataset = DataLoader(val_dataset,
                     batch_size=BATCH_SIZE, 
                     drop_last=False,
                     shuffle=True)
    # Models
    model = Seq2Seq(config,vocab_inp_size,vocab_tar_size)

    if config['gpu']:
        model = model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))

    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_normal_(param)
    print("Weight Initialized")

    for epoch in range(FLAGS.epochs):
        run_state = (epoch, FLAGS.epochs)

        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
        model, optimizer = train(model, optimizer, train_dataset, run_state)
        metrics = evaluate(model, eval_dataset)

    # TODO implement save models function

    # TODO to record model training loss changes 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    FLAGS, _ = parser.parse_known_args()
    print(FLAGS)
    run()


