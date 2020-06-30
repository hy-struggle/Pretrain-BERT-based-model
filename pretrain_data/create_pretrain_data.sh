#!/usr/bin/env bash


python create_pretrain_data.py --do_whole_word_mask --remove_nsp --do_ngram --reduce_memory \
--epochs_to_generate=10