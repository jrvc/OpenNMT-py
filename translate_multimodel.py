from translate import *
import argparse, onmt

from ipdb import launch_ipdb_on_exception

if __name__ == "__main__":
    parser = ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    # only this line is extra to normal translate.py
    onmt.opts.translate_multimodel(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    with launch_ipdb_on_exception():
        main(opt)
    #main(opt)
