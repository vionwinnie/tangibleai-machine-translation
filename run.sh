#!/bin/sh


# Set Paths for arguments
homeDir=$HOME
dataPath=${homeDir}/spa.txt
modelCheckPointPath="${homeDir}/export_model"
metricsPath="${homeDir}/metrics"
config='seq2seq.json'
n_epoch=20

# Change to repo directory
codeDir="${homeDir}/tangiblemt"
cd ${codeDir}

echo "$config" "$n_epoch" "$dataPath" "$modelCheckPointPath" "$metricsPath"

# Run Script
python main.py --config seq2seq.json --epochs 20 --data_path /home/winnie/spa.txt --model_checkpoint_dir /home/winnie/export_model --metrics_dir /home/winnie/metrics

python main.py --config "$config" -- epoch "$n_epoch" --data_path "$dataPath --model_checkpoint_dir "$modelCheckPointPath" --metrics_dir "$metricsPath"
