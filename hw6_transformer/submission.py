from cfg import config
import os
from utils import try_load_checkpoint
from valid_script import validate
import logging
from model import build_model_transformer as build_model, medium_arch_args as arch_args
from loss import LabelSmoothedCrossEntropyCriterion
from data import load_task, load_data_iterator
from tqdm import tqdm
import torch
from fairseq import utils
from inference_script import inference_step

# CUDA環境
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate_prediction(model, task, split="test", outfile="./prediction.txt", sequence_generator=None):    
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    idxs = []
    hyps = []

    model.eval()
    progress = tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)

            # 進行推論
            s, h, r = inference_step(sample, model, sequence_generator, task)
            
            hyps.extend(h)
            idxs.extend(list(sample['id']))
            
    # 根據 preprocess 時的順序排列
    hyps = [x for _,x in sorted(zip(idxs,hyps))]
    
    with open(outfile, "w") as f:
        for h in hyps:
            f.write(h+"\n")

if __name__ == '__main__':
    proj = "hw5.seq2seq"
    logger = logging.getLogger(proj)

    # 把幾個 checkpoint 平均起來可以達到 ensemble 的效果
    checkdir=config.savedir
    os.system(f"{config.python_path} ./fairseq/scripts/average_checkpoints.py \
    --inputs {checkdir} \
    --num-epoch-checkpoints 5 \
    --output {checkdir}/avg_last_5_checkpoint.pt")

    task = load_task(logger)

    model = build_model(arch_args, task)
    logger.info(model)

    sequence_generator = task.build_generator([model], config)

    criterion = LabelSmoothedCrossEntropyCriterion(
        smoothing=0.1,
        ignore_index=task.target_dictionary.pad(),
    )

    model = model.to(device=device)
    criterion = criterion.to(device=device)

    # checkpoint_last.pt : 最後一次檢驗的檔案
    # checkpoint_best.pt : 檢驗 BLEU 最高的檔案
    # avg_last_5_checkpoint.pt:　最5後個檔案平均
    try_load_checkpoint(model, name="avg_last_5_checkpoint.pt", logger=logger)
    validate(model, task, criterion, sequence_generator, logger, log_to_wandb=False)
    generate_prediction(model, task, sequence_generator=sequence_generator)

