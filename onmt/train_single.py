#!/usr/bin/env python
"""
    Training on a single process
"""
from __future__ import division

import argparse
import os
import random
import torch
import torch.nn as nn

import onmt.opts as opts

from collections import defaultdict, OrderedDict

from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features
from onmt.model_builder import build_model, build_embeddings_then_encoder
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_postprocessing(opt):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    opt.brnn = (opt.encoder_type == "brnn")

    if opt.rnn_type == "SRU" and not opt.gpuid:
        raise AssertionError("Using SRU requires -gpuid set.")

    if torch.cuda.is_available() and not opt.gpuid:
        logger.info("WARNING: You have a CUDA device, should run with -gpuid")

    if opt.gpuid:
        torch.cuda.set_device(opt.device_id)
        if opt.seed > 0:
            # this one is needed for torchtext random call (shuffled iterator)
            # in multi gpu it ensures datasets are read in the same order
            random.seed(opt.seed)
            # These ensure same initialization in multi gpu mode
            torch.manual_seed(opt.seed)
            torch.cuda.manual_seed(opt.seed)

    return opt


def build_train_iter_fct(path_, fields_, opt_):

    def train_iter_wrapper():
        return build_dataset_iter(lazily_load_dataset("train", path_),
                                  fields_,
                                  opt_)

    return train_iter_wrapper



def main(opt):
    opt = training_opt_postprocessing(opt)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
    else:
        checkpoint = None
        model_opt = opt

    # For each dataset, load fields generated from preprocess phase.
    train_iter_fcts = {}

    # Loop to create encoders
    encoders = OrderedDict()
    for (src_tgt_lang), data_path in zip(opt.src_tgt, opt.data):
        src_lang, tgt_lang = src_tgt_lang.split('-')
        # Peek the first dataset to determine the data_type.
        # (All datasets have the same data_type).
        first_dataset = next(lazily_load_dataset("train", data_path))
        data_type = first_dataset.data_type
        fields = _load_fields(first_dataset, data_type, data_path, checkpoint)

        # Report src/tgt features.
        src_features, tgt_features = _collect_report_features(fields)
        for j, feat in enumerate(src_features):
            logger.info(' * src feature %d size = %d'
                        % (j, len(fields[feat].vocab)))
        for j, feat in enumerate(tgt_features):
            logger.info(' * tgt feature %d size = %d'
                        % (j, len(fields[feat].vocab)))

        # Build model.
        encoder = build_embeddings_then_encoder(model_opt, fields)

        encoders[src_lang] = encoder

        def valid_iter_fct(): return build_dataset_iter(
            lazily_load_dataset("valid", data_path), fields, opt)

        # add this dataset iterator to the training iterators
        train_iter_fcts[(src_lang, tgt_lang)] = build_train_iter_fct(data_path,
                                                                     fields,
                                                                     opt)


    # build the model with all of the encoders and all of the decoders
    # note here we just replace the encoders of the final model
    model = build_model(model_opt, opt, fields, checkpoint)

    # TODO: this is a hack which will only work for multi-encoder
    encoder_ids = {lang_code: idx
                   for lang_code, idx
                   in zip(encoders.keys(), range(len(list(encoders.keys()))))}
    encoders = nn.ModuleList(encoders.values())
    model.encoder_ids = encoder_ids
    model.encoders = encoders

    if len(opt.gpuid) > 0:
        model.to('cuda')

    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    logger.info(model)

    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, model, fields, optim, data_type, model_saver=model_saver)

    #trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps,
    #              opt.valid_steps)
    trainer.train(train_iter_fcts, valid_iter_fct, opt.train_steps,
                  opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
