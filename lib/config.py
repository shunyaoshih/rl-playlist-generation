"""Parameters definition"""

import logging
import argparse

from lib.utils import num_of_lines

def params_setup():
    """Parameters definition"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='pretrain', help='')
    parser.add_argument('--debug', type=int, default=0, help='')
    parser.add_argument('--file_output', type=int, default=0, help='')
    parser.add_argument('--steps', type=int, default=500000, help='')

    parser.add_argument('--learning_rate', type=float, default=0.005, help='')
    parser.add_argument('--rl_learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--init_weight', type=float, default=0.1, help='')
    parser.add_argument('--max_gradient_norm', type=float, default=5.0, help='')
    parser.add_argument('--num_units', type=int, default=64, help='')
    parser.add_argument('--num_layers', type=int, default=4, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--vocab_size', type=int, default=-1, help='')
    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--max_len', type=int, default=30, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')

    # mechanism
    parser.add_argument('--concatenate', type=int, default=1, help='')
    parser.add_argument('--attention', type=int, default=1, help='')
    parser.add_argument('--attention_window', type=int, default=4, help='')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', help='')
    parser.add_argument("--start_decay_step", type=int, default=5000, help='')
    parser.add_argument('--decay_steps', type=int, default=1000, help='')
    parser.add_argument('--decay_factor', type=float, default=0.98, help='')
    parser.add_argument('--steps_per_stats', type=int, default=1000, help='')
    parser.add_argument('--model_dir', type=str, default='model', help='')
    parser.add_argument('--rl_model_dir', type=str, default='rl_model', help='')
    parser.add_argument('--original_model_dir', type=str, default='model', help='')

    parser.add_argument('--distance', type=float, default=1.2, help='')

    # reward
    parser.add_argument('--dist', type=float, default=0.0, help='')
    parser.add_argument('--pop', type=float, default=0.0, help='')
    parser.add_argument('--date', type=float, default=0.0, help='')
    parser.add_argument('--probs', type=float, default=0.0, help='')

    para = parser.parse_args()

    para.vocab_size = num_of_lines('./data/vocab_default.txt')
    para.num_units = para.embedding_size
    para.logging_level = logging.INFO

    if para.debug == 1:
        para.logging_level = logging.DEBUG
        para.embedding_size = 3
        para.num_units = 3
        para.num_layers = 1
        para.max_len = 5
        para.batch_size = 1

    if para.mode == 'rl':
        para.dropout = 0.0

    if para.mode == 'test':
        para.dropout = 0.0
        para.batch_size = num_of_lines('./results/in.txt')

    return para
