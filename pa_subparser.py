import argparse

def add_pa_subparser(subparsers):
    subparser = subparsers.add_parser('pa')          
    subparser.add_argument('--node_mode', type=int, default=0)  # 0: feature onehot