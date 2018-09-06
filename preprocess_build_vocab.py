
from preprocess import build_save_vocab
from onmt.utils.logging import init_logger, logger
import onmt.inputters as inputters
import argparse
import onmt.opts as opts

import torch
import os


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('-data_type', default="text",
                       help="""Type of the source input.
                       Options are [text|img].""")

    group.add_argument('-train_dataset_prefixes', nargs='+', help="""List of prefixes for data.train.X.pt files to create vocabulary from.
    The vocabulary will be stored to data.vocab.pt.""")

    group.add_argument('-num_src_features', default=1, help="""Number of src features. Default: 1.""")
    group.add_argument('-num_tgt_features', default=1, help="""Number of src features. Default: 1.""")


    # Dictionary options, for text corpus

    group = parser.add_argument_group('Vocab')
    group.add_argument('-src_vocab', default="",
                       help="""Path to an existing source vocabulary. Format:
                       one word per line.""")
    group.add_argument('-tgt_vocab', default="",
                       help="""Path to an existing target vocabulary. Format:
                       one word per line.""")
    group.add_argument('-features_vocabs_prefix', type=str, default='',
                       help="Path prefix to existing features vocabularies")
    group.add_argument('-src_vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")
    group.add_argument('-tgt_vocab_size', type=int, default=50000,
                       help="Size of the target vocabulary")

    group.add_argument('-src_words_min_frequency', type=int, default=0)
    group.add_argument('-tgt_words_min_frequency', type=int, default=0)

    group.add_argument('-dynamic_dict', action='store_true',
                       help="Create dynamic dictionaries")
    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add_argument('-src_seq_length', type=int, default=50,
                       help="Maximum source sequence length")
    group.add_argument('-src_seq_length_trunc', type=int, default=0,
                       help="Truncate source sequence length.")
    group.add_argument('-tgt_seq_length', type=int, default=50,
                       help="Maximum target sequence length to keep.")
    group.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                       help="Truncate target sequence length.")
    group.add_argument('-lower', action='store_true', help='lowercase data')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('-shuffle', type=int, default=1,
                       help="Shuffle data")
    group.add_argument('-seed', type=int, default=3435,
                       help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=100000,
                       help="Report status every this many sentences")
    group.add_argument('-log_file', type=str, default="",
                       help="Output logs to a file under this path.")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help="Window size for spectrogram in seconds.")
    group.add_argument('-window_stride', type=float, default=.01,
                       help="Window stride for spectrogram in seconds.")
    group.add_argument('-window', default='hamming',
                       help="Window type for spectrogram generation.")

def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess_build_vocab.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    preprocess_opts(parser)

    opt = parser.parse_args()
   # torch.manual_seed(opt.seed)

   # check_existing_pt_files(opt)

    return opt


# Note: This code will be good enough for shared vocabulary for all decoders and encoders.

def main():
    opt = parse_args()
    init_logger(opt.log_file)

    logger.info("Building & saving vocabulary...")

    logger.info("Extracting features...")

    src_nfeats = opt.num_src_features
    tgt_nfeats = opt.num_tgt_features
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)

    train_dataset_pref = opt.train_dataset_prefixes

    train_dataset_files = []
    vocab_files = []
    for pref in train_dataset_pref:
        basedir = os.path.dirname(pref)
        pref_basename = os.path.basename(pref)+".train."
        vocab_files.append(pref+".vocab.pt")
        for fn in os.listdir(basedir):
            if fn.startswith(pref_basename):
                ptfile = os.path.join(basedir, fn)
                train_dataset_files.append(ptfile)

   # train_dataset_files =
   # print(train_dataset_files)
    #build_save_vocab(train_dataset_files, fields, opt)
    """ Building and saving the vocab """
    fields = inputters.build_vocab(train_dataset_files, fields, opt.data_type,
                                   opt.share_vocab,
                                   opt.src_vocab,
                                   opt.src_vocab_size,
                                   opt.src_words_min_frequency,
                                   opt.tgt_vocab,
                                   opt.tgt_vocab_size,
                                   opt.tgt_words_min_frequency)

    # Can't save fields, so remove/reconstruct at training time.
    for vocab_file in vocab_files:
        #vocab_file = opt.save_data + '.vocab.pt'
        logger.info("saving vocab to %s" % vocab_file)
        torch.save(inputters.save_fields_to_vocab(fields), vocab_file)






if __name__ == "__main__":
    main()