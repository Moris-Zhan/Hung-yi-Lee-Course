import logging
import sys
from pathlib import Path
from cfg import config

import sys
import pdb
import pprint
import logging

import torch
from tqdm import tqdm_notebook as tqdm
from pathlib import Path
from fairseq import utils
from data import load_task, load_data_iterator

from model import build_model_transformer as build_model, medium_arch_args as arch_args
from loss import LabelSmoothedCrossEntropyCriterion
from optimizer import NoamOpt
from utils import try_load_checkpoint, validate_and_save
from train_script import train_one_epoch

# CUDA環境
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO", # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )
    proj = "hw5.seq2seq-bt"
    logger = logging.getLogger(proj)
    if config.use_wandb:
        import wandb
        wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

    ## setup task
    task = load_task(logger)

    if config.use_wandb:
        wandb.config.update(vars(arch_args))

    model = build_model(arch_args, task)
    logger.info(model)

    criterion = LabelSmoothedCrossEntropyCriterion(
        smoothing=0.1,
        ignore_index=task.target_dictionary.pad(),
    )

    optimizer = NoamOpt(
        model_size=arch_args.encoder_embed_dim, 
        factor=config.lr_factor, 
        warmup=config.lr_warmup, 
        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))

    # fairseq 的 beam search generator
    # 給定模型和輸入序列，用 beam search 生成翻譯結果
    sequence_generator = task.build_generator([model], config)

    model = model.to(device=device)
    criterion = criterion.to(device=device)

    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("encoder: {}".format(model.encoder.__class__.__name__))
    logger.info("decoder: {}".format(model.decoder.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info("optimizer: {}".format(optimizer.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")

    epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)
    try_load_checkpoint(model, optimizer, name=config.resume, logger=logger)
    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # train for one epoch
        train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps, logger=logger)
        stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch, sequence_generator=sequence_generator, logger=logger)
        logger.info("end of epoch {}".format(epoch_itr.epoch))    
        epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)