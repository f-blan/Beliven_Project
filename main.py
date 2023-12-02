import sys
from argument_parser import parse_arguments
from data_analysis import data_analysis
from train import train

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    if args.mode == "data_analysis":
        data_analysis(args)
    elif args.mode== "train":
        train(args)