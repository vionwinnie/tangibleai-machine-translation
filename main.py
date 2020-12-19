import argparse
import torch
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime


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
    config_path = os.path.join(dir_path,"experiment", FLAGS.config)

    if not os.path.exists(config_path):
        raise FileNotFoundError

    if not os.path.exists(FLAGS.data_path):
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
    train_dataset = data.DataLoader(train_dataset,
                     batch_size=config['batch_size'], 
                     drop_last=True,
                     shuffle=True)

    eval_dataset = data.DataLoader(val_dataset,
                     batch_size=config['batch_size'], 
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

    ## Train and Evaluate over epochs
    all_train_avg_loss = []
    all_eval_avg_loss = []
    all_eval_avg_acc = []

    for epoch in range(FLAGS.epochs):
        run_state = (epoch, FLAGS.epochs)

        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
        model, optimizer,train_avg_loss = train(
                model, 
                optimizer, 
                train_dataset, 
                run_state,
                config['debug'])
        all_train_avg_loss.append(train_avg_loss)

        # Return Val Set Loss and Accuracy
        eval_avg_loss, eval_acc = evaluate(
                model, 
                eval_dataset,
                targ_index,
                config['debug'])
        all_eval_avg_loss.append(eval_avg_loss)
        all_eval_avg_acc.append(eval_acc)

        # Save Model Checkpoint
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': eval_avg_loss,
            } 
        
        checkpoint_path = '{}/epoch_{:0.0f}_val_loss_{:0.3f}.pt'.format(FLAGS.model_checkpoint_dir,epoch,eval_avg_loss)
        torch.save(checkpoint_dict,checkpoint_path)

    # Export Model Learning Curve Info
    df = pd.DataFrame({
        'epoch':range(FLAGS.epochs),
        'train_loss': all_train_avg_loss,
        'eval_loss': all_eval_avg_loss,
        'eval_acc': all_eval_avg_acc
        })

    now = datetime.now() 
    current_time = now.strftime("%Y%m%d%H%M%S")
    export_path = '{}/{}_{:0.0f}_bz_{}_val_loss_{:0.3f}.csv'.format(FLAGS.metrics_dir,current_time,FLAGS.epochs,config['batch_size'],eval_avg_loss)
    df.to_csv(export_path,index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--data_path', type=Path)
    parser.add_argument('--model_checkpoint_dir',type=Path)
    parser.add_argument('--metrics_dir',type=Path)
    FLAGS, _ = parser.parse_known_args()
    run()


