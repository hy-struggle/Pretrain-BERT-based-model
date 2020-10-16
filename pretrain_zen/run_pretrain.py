# /usr/bin/env python
# coding=utf-8
import time
import datetime
from tqdm import trange
import utils
from argparse import ArgumentParser
from pathlib import Path
import os
import logging
from utils import set_logger, Params
from ZEN import WEIGHTS_NAME, CONFIG_NAME
import torch
import json
import numpy as np
import random
from ZEN import BertTokenizer, ZenForPreTraining, ZenConfig
from ZEN import WarmupLinearSchedule, BertAdam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from dataloader import PregeneratedDataset

parser = ArgumentParser()
parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument('--multi_gpu', action='store_true', help='是否使用多GPU')
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--scratch', action='store_true', help="Whether to train from scratch")
parser.add_argument('--save_name', type=str, default="zen",
                    help="The prefix used for saving the remote model")
parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--train_batch_size", default=256, type=int,
                    help="Total batch size for training.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")

parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True."
                         "0 (default value): dynamic loss scaling."
                         "Positive power of 2: static loss scaling value.")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def train(model, train_dataloader, epoch, optimizer, params, global_step, warmup_schedule=None):
    """
    :param epoch: current epoch
    :param global_step: current total step
    :param warmup_schedule: for fp16 warmup
    :return:
    """
    model.train()
    # 记录平均损失
    loss_avg = utils.RunningAverage()

    t = trange(len(train_dataloader), desc=f"Epoch {epoch}")
    for step, _ in enumerate(t):
        batch = next(iter(train_dataloader))
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, segment_ids, lm_label_ids, is_next, ngram_ids, ngram_masks, ngram_positions, \
        ngram_starts, ngram_lengths, ngram_segment_ids = batch

        loss = model(input_ids,
                     ngram_ids,
                     ngram_positions,
                     segment_ids,
                     ngram_segment_ids,
                     input_mask,
                     ngram_masks,
                     lm_label_ids,
                     is_next)

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        loss_avg.update(loss.item() * args.gradient_accumulation_steps)
        t.set_postfix_str(f"Loss: {loss_avg():.5f}")

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = args.learning_rate * warmup_schedule.get_lr(step=global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    # Save a trained model
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')
    # dir to save
    saving_path = Path(os.path.join(params.pretrain_model_dir, args.save_name + st + "_epoch_" + str(epoch)))
    if saving_path.is_dir() and list(saving_path.iterdir()):
        logging.warning(f"Output directory ({saving_path}) already exists and is not empty!")
    saving_path.mkdir(parents=True, exist_ok=True)

    logging.info("***** Saving fine-tuned model ***** ")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    output_model_file = os.path.join(saving_path, WEIGHTS_NAME)
    output_config_file = os.path.join(saving_path, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)


if __name__ == '__main__':
    params = Params()
    set_logger(save=True, log_path=params.pretrain_model_dir)
    # get data epochs
    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = params.pretrain_data_dir / f"epoch_{i}.json"
        metrics_file = params.pretrain_data_dir / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs

    if args.local_rank == -1 and args.multi_gpu:
        params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        params.device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        params.device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # 实际的batch_size，用了梯度累加的结果
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=params.corpus_dir,
                                              do_lower_case=False)

    # get total steps
    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]
    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.scratch:
        config = ZenConfig(vocab_size_or_config_json_file=1446,
                           word_vocab_size=1979,
                           hidden_size=768,
                           num_hidden_layers=6,
                           num_attention_heads=12,
                           intermediate_size=3072,
                           hidden_act="gelu",
                           hidden_dropout_prob=0.1,
                           attention_probs_dropout_prob=0.1,
                           max_position_embeddings=512,
                           type_vocab_size=2,
                           initializer_range=0.02,
                           layer_norm_eps=1e-12,
                           num_hidden_word_layers=3)
        model = ZenForPreTraining(config)
    else:
        model = ZenForPreTraining.from_pretrained(params.bert_model_dir)

    if args.fp16:
        model.half()
    model.to(params.device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    global_step = 0
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", total_train_examples)
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    for epoch in range(args.epochs):
        # get dataloader
        epoch_dataset = PregeneratedDataset(epoch=epoch,
                                            training_path=params.pretrain_data_dir,
                                            tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs,
                                            do_ngram=True,
                                            fp16=args.fp16)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        train(model, train_dataloader, epoch, optimizer, params, global_step, warmup_schedule=None)
