"""Parse raw_data.txt"""

import sys
from collections import defaultdict
from tqdm import tqdm

MAX_LEN = int(sys.argv[1])
VOCAB_SIZE = int(sys.argv[2])

def read_vocabulary_file():
    """Read the vocabulary dictionary"""
    vocab_list = open('./vocab_default.txt', 'r').read().splitlines()
    vocab_dct = defaultdict(lambda: '3', [[word, str(i)] for i, word in enumerate(vocab_list)])
    return vocab_dct

def parse_file(raw_seqs):
    """raw/raw_dat.txt to ./train_ids.txt"""
    with open('./train_raw_ids.txt', 'w') as output_file:
        for i in tqdm(range(len(raw_seqs))):
            seq = raw_seqs[i]
            for final_song in range(1, len(seq)):
                output_file.write(' '.join(seq[0:final_song + 1]) + '\n')

    dct = read_vocabulary_file()
    raw_seqs = [[dct[word] for word in seq] for seq in raw_seqs]

    with open('./train_ids.txt', 'w') as output_file:
        for i in tqdm(range(len(raw_seqs))):
            seq = raw_seqs[i]
            for final_song in range(1, len(seq)):
                output_file.write(' '.join(seq[0:final_song + 1]) + '\n')

def create_vocab_file(raw_seqs):
    """Create vocab file for output"""
    start_vocab = ['_PAD', '_BOS', '_EOS', '_UNK']
    vocab = {}
    for seq in raw_seqs:
        for word in seq:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab_list = sorted(vocab.items(), key=lambda t: t[::-1], reverse=True)
    vocab_list = [vocab_list[i][0] for i in range(min(len(vocab_list), VOCAB_SIZE - 4))]
    vocab_list = start_vocab + vocab_list
    print('vocab size: {}'.format(len(vocab_list)))

    with open('./vocab_default.txt', 'w') as output_file:
        output_file.write('\n'.join(vocab_list))

if __name__ == '__main__':
    print('parse raw_dat.txt')
    RAW_SEQS = open('./raw/raw_data.txt', 'r').read().splitlines()
    RAW_SEQS = [seq.strip().split(' ') for seq in RAW_SEQS]
    RAW_SEQS = [seq[1:] for seq in RAW_SEQS]

    create_vocab_file(RAW_SEQS)

    DCT = read_vocabulary_file()
    SEQS = [[word for word in seq if DCT[word] != '3'] for seq in RAW_SEQS]
    SEQS = [seq[0:MAX_LEN] + ['_EOS'] for seq in SEQS]

    parse_file(SEQS)
