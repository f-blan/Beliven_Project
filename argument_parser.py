import argparse
from argparse import Namespace
from typing import List
import os

def parse_arguments(args_list: List[str]) -> Namespace:
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    

    #control which script to launch
    parser.add_argument("--mode", choices=["data_analysis", "train"], default = "data_analysis", help="select the functionality to run")

    #training arguments
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for training/testing")
    parser.add_argument("--crop_size", default = 256, type=int, help= "crop size before feeding the imgs to the networks")
    parser.add_argument("--n_epochs", default = 12, type=int, help="number of epochs for training")
    parser.add_argument("--lr", type=float, default = 0.0001, help="initial value for learning rate")
    parser.add_argument("--use_augmentation", action="store_true", help=("apply data augmentation during training"))
    parser.add_argument("--model", choices=["curstom", "resnet"], default="resnet", help="choose classifier model")

    parser.add_argument("--ds_folder", default = os.path.join(".", "dataset"), help="folder containing the dataset (should be already divided into training and testing)")


    args = parser.parse_args(args_list)

    args.train_dir = os.path.join(args.ds_folder, "train")
    args.test_dir = os.path.join(args.ds_folder, "validation")
    
    return args