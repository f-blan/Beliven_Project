import argparse
from argparse import Namespace
from typing import List
import os

def parse_arguments(args_list: List[str]) -> Namespace:
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #general arguments
    parser.add_argument("--app", choices=["checkers"], default="checkers", help="specify which RL application to run")
    parser.add_argument("--log_folder", default=os.path.join(".", "logs"))

    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--crop_size", default = 256)
    parser.add_argument("--ds_folder", default = os.path.join(".", "ds"))


    args = parser.parse_args(args_list)

    args.train_dir = os.path.join(args.ds_folder, "train")
    args.test_dir = os.path.join(args.ds_folder, "validation")
