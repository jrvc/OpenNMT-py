
from preprocess import build_save_vocab
from onmt.utils.logging import init_logger, logger
from collections import Counter
import onmt.inputters as inputters
import argparse
import onmt.opts as opts

import torch
import torchtext
import os

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

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
    
def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = f.vocab.stoi
            vocab.append((k, f.vocab))
    return vocab

def AddCounter(lang_tuple, lang, dictLangFreqs):
    Vocab = lang_tuple[1]
    freqs = Vocab.freqs
    c = dictLangFreqs[lang]
    c += freqs
    dictLangFreqs[lang] = c

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

    train_dataset_pref = opt.train_dataset_prefixes[0]

    train_dataset_files = []
    vocab_files = []
    """
    for pref in train_dataset_pref:
        basedir = os.path.dirname(pref)
        pref_basename = os.path.basename(pref)+".train."
        vocab_files.append(pref+".vocab.pt")
        for fn in os.listdir(basedir):
            if fn.startswith(pref_basename):
                ptfile = os.path.join(basedir, fn)
                train_dataset_files.append(ptfile)
    """
    vocabfiles = [f for f in os.listdir(train_dataset_pref) if f.find('vocab.pt')>-1]
    langpairs = [vf[vf.find('.')+1:vf.find('.vocab')] for vf in vocabfiles]
    langs = set([langpair.split('-')[1] for langpair in langpairs])
    dictLangFreqs = {}
    for lang in langs:
        dictLangFreqs[lang] = Counter()
    
    # update frequencies to take into account all the files
    for i in range(len(langpairs)):
          vocab_object = torch.load(train_dataset_pref+'/'+vocabfiles[i])
          src_lang, tgt_lang = langpairs[i].split('-')
          AddCounter(vocab_object[0], src_lang, dictLangFreqs)
          AddCounter(vocab_object[1], tgt_lang, dictLangFreqs)

    """ saving the new vocab """
    for i in range(len(langpairs)):
        langSRC, langTGT = langpairs[i].split('-')
        #fields = {}

        #fields["src"] = torchtext.data.Field(
        #        include_lengths=True)
        #fields["tgt"] = torchtext.data.Field(
        #        pad_token=PAD_WORD)

        c_tgt = dictLangFreqs[langTGT]
        vocab_tgt = torchtext.vocab.Vocab(c_tgt,
                                  specials=[UNK_WORD, PAD_WORD,
                                            BOS_WORD, EOS_WORD],
                                  max_size=50000)
        fields["tgt"].vocab = vocab_tgt

        c_src = dictLangFreqs[langSRC]
        vocab_src = torchtext.vocab.Vocab(c_src,
                                  specials=[UNK_WORD, PAD_WORD,
                                            BOS_WORD, EOS_WORD],
                                  max_size=50000)
        fields["src"].vocab = vocab_src

        vocab_file = train_dataset_pref+'/'+vocabfiles[i]
        torch.save(save_fields_to_vocab(fields), vocab_file)

        

if __name__ == "__main__":
    main()
