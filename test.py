import pandas as pd
import json
import os
import argparse
from pathlib import Path

def run():
    print(FLAGS)

    ## Load Config from JSON file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("directory path------------",dir_path)
    config_path = os.path.join(dir_path,"experiment", FLAGS.config)

    if not os.path.exists(config_path):
        print(config_path)
        raise FileNotFoundError

    with open(config_path, "r") as f:
        config = json.load(f)

    if not os.path.exists(FLAGS.data_path):
        print(FLAGS.data_path)
        raise FileNotFoundError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path)
    #parser.add_argument('--data_path', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    FLAGS, _ = parser.parse_known_args()
    run()
