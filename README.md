# tangibleai-machine-translation
Modularized code for Tangible AI Machine Translation Project

## How to run the code
- Shell script: Update paths inside ```run.sh``` and run the shell script ```./run.sh```
- Run Python script ```main.py``` 
- ``` python main.py --config ${model_hyperparameter_json} --epochs ${num_epoch} --data_path ${training_file} --model_checkpoint_dir ${export_path} --metrics_dir ${metrics_path}```

- For example:
``` python main.py --config seq2seq.json -- epoch 20 --data_path \home\machine-translation\data\spa.txt --model_checkpoint_dir \home\machine-translation --metrics_dir \home\machine-translation\metrics ```

- Model Hyperparameter Json: Name of the config file (under the experiment subdirectory)
- Epoch: Number of Epoch
- Training Text File: Directory of the training corpus (.txt)
- Model Checkpoint Path: Directory to save model checkpoint
- Metric Directory: Directory to save learning curve and model metrics

## Dependencies:
- Python 3.7.8
- Pytorch 1.6

## Packages:
- NLTK
- editdistance
