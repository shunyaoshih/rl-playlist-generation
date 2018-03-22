"""The procedure of testing"""

from lib.utils import read_all_embedding
from lib.model_utils import predict

def test(para, sess, model):
    """The procedure of testing"""
    embed_dct = read_all_embedding()
    seed_ids = open('./results/in.txt', 'r').read().splitlines()
    rev_vocab = open('./data/vocab_default.txt', 'r').read().splitlines()

    output_file = open('./results/out.txt', 'w')
    _, results = predict(rev_vocab, embed_dct, para, sess, model, seed_ids)
    for seq in results:
        output_file.write(' '.join(seq) + '\n')
    output_file.close()
