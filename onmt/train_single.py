#!/usr/bin/env python
"""Training on a single process."""
import os

import torch

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab
from onmt.model_builder import build_model, build_embeddings_then_encoder, \
    build_decoder_and_generator
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

from collections import OrderedDict

def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        #vocab = torch.load(opt.data + '.vocab.pt')

    train_iters = OrderedDict()
    valid_iters = OrderedDict()

    encoders = OrderedDict()
    decoders = OrderedDict()

    generators = OrderedDict()
    src_vocabs = OrderedDict()
    tgt_vocabs = OrderedDict()

    # variables needed for sharing the same embedding matrix across encoders and decoders
    firstTime=True
    weightToShare=None

    # we share the word embedding space when source lang and target lang are the same
    mapLang2Emb = {}

    for (src_tgt_lang), data_path in zip(opt.src_tgt, opt.data):
        src_lang, tgt_lang = src_tgt_lang.split('-')

        vocab = torch.load(data_path + '.vocab.pt')

        # check for code where vocab is saved instead of fields
        # (in the future this will be done in a smarter way)
        if old_style_vocab(vocab):
            fields = load_old_vocab(
                vocab, opt.model_type, dynamic_dict=opt.copy_attn)
        else:
            fields = vocab

        # Report src and tgt vocab sizes, including for features
        for side in ['src', 'tgt']:
            f = fields[side]
            try:
                f_iter = iter(f)
            except TypeError:
                f_iter = [(side, f)]
            for sn, sf in f_iter:
                if sf.use_vocab:
                    logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

        # Build model.
        encoder, src_embeddings = build_embeddings_then_encoder(model_opt, fields)

        encoders[src_lang] = encoder

        decoder, generator, tgt_embeddings = build_decoder_and_generator(model_opt, fields)

        decoders[tgt_lang] = decoder

        # Share the embedding matrix across all the encoders and decoders - preprocess with share_vocab required.
        if model_opt.share_embeddings and firstTime:
                tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight
                weightToShare = src_embeddings.word_lut.weight
        if model_opt.share_embeddings and (not firstTime):
                tgt_embeddings.word_lut.weight = weightToShare
                src_embeddings.word_lut.weight = weightToShare
        firstTime = False

        if src_lang in mapLang2Emb:
                encoder.embeddings.word_lut.weight = mapLang2Emb.get(src_lang)
        else:
                mapLang2Emb[src_lang] = src_embeddings.word_lut.weight
        if tgt_lang in mapLang2Emb:
                decoder.embeddings.word_lut.weight = mapLang2Emb.get(tgt_lang)
        else:
            mapLang2Emb[tgt_lang] = tgt_embeddings.word_lut.weight

        src_vocabs[src_lang] = fields['src'].base_field.vocab
        tgt_vocabs[tgt_lang] = fields['tgt'].base_field.vocab

        generators[tgt_lang] = generator

        # add this dataset iterator to the training iterators
        train_iters[(src_lang, tgt_lang)] = build_dataset_iter('train',
                                                                fields,
                                                                data_path,
                                                                opt)
        # add this dataset iterator to the validation iterators
        valid_iters[(src_lang, tgt_lang)] = build_dataset_iter('valid',
                                                                fields,
                                                                data_path,
                                                                opt,
                                                                is_train=False)


    # Build model.
    model = build_model(model_opt, opt, fields, encoders, decoders,
            generators, src_vocabs, tgt_vocabs, checkpoint)

    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, generators, tgt_vocabs,
        model_saver=model_saver)

    #train_iter = build_dataset_iter("train", fields, opt)
    #valid_iter = build_dataset_iter(
    #    "valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0
    trainer.train(
        train_iters,
        train_steps,
        opt.save_checkpoint_steps,
        valid_iters,
        opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()
