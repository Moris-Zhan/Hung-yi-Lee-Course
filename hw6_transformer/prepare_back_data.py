import re
import logging
from pathlib import Path
import os
import random
import sentencepiece as spm
# fairseq_cli
from fairseq import utils
import subprocess
from cfg import config
from submission import generate_prediction
from data import load_task
from model import build_model_transformer as build_model, medium_arch_args as arch_args
import torch

# CUDA環境
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir = './DATA/rawdata'
mono_dataset_name = 'mono'

def download_data(): 
    mono_prefix = Path(data_dir).absolute() / mono_dataset_name
    mono_prefix.mkdir(parents=True, exist_ok=True)

    urls = (
        # '"https://onedrive.live.com/download?cid=3E549F3B24B238B4&resid=3E549F3B24B238B4%214986&authkey=AANUKbGfZx0kM80"',
    # If the above links die, use the following instead. 
        "https://www.csie.ntu.edu.tw/~r09922057/ML2021-hw5/ted_zh_corpus.deduped.gz",
    # # If the above links die, use the following instead. 
    #     "https://mega.nz/#!vMNnDShR!4eHDxzlpzIpdpeQTD-htatU_C7QwcBTwGDaSeBqH534",
    )
    file_names = (
        'ted_zh_corpus.deduped.gz',
    )

    for u, f in zip(urls, file_names):
        path = mono_prefix/f
        if not path.exists():
            if 'mega' in u:
                os.system(f"megadl {u} --path {path}")
            else:
                os.system(f"wget {u} -O {path}")
        else:
            print(f'{f} is exist, skip downloading')
        if path.suffix == ".tgz":
            os.system(f"tar -xvf {path} -C {mono_prefix}")
        elif path.suffix == ".zip":
            os.system(f"unzip -o {path} -d {mono_prefix}")
        elif path.suffix == ".gz":
            os.system(f"gzip -fkd {path}")

    return mono_prefix

def strQ2B(ustring):
    """把字串全形轉半形"""
    # 參考來源:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全形空格直接轉換
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全形字元（除空格）根據關係轉化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)

def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
        s = s.replace('-', '') # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s) # keep punctuation
    elif lang == 'zh':
        s = strQ2B(s) # Q2B
        s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s) # keep punctuation
    s = ' '.join(s.strip().split())
    return s

def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())

def clean_corpus(mono_prefix, src_lang, ratio=9, max_len=1000, min_len=1):
    with open(f'{mono_prefix}/ted_zh_corpus.clean', 'w') as out_f:
        with open(f'{mono_prefix}/ted_zh_corpus.deduped', 'r') as f:
            for line in f.readlines():
                line = line.strip()       
                s1 = clean_s(line, src_lang)
                s1_len = len_s(s1, src_lang)
                if min_len > 0: # remove short sentence
                    if s1_len < min_len:
                        continue
                if max_len > 0: # remove long sentence
                    if s1_len > max_len:
                        continue                   
                print(s1, file=out_f)

def subword_units(mono_prefix, src_lang, tgt_lang):
    vocab_size = 8000

    data_dir = "/home/leyan/Documents/Hung-yi-Lee/hw6_transformer/DATA/rawdata/"
    prefix = Path(data_dir).absolute() / "ted2020"

    spm_model = spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))

    # already process 
    for lang in [src_lang]:
        out_path = mono_prefix/f'mono.tok.{lang}'
        # os.system(f"rm {out_path}")
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(mono_prefix/f'mono.tok.{lang}', 'w') as out_f:
                with open(mono_prefix/f'ted_zh_corpus.clean', 'r') as in_f:     
                    lines = in_f.readlines()             
                    for line in lines:
                        line = line.strip()
                        tok = spm_model.encode(line, out_type=str)
                        print(' '.join(tok), file=out_f)
                    line_counts = len(lines)

                with open(f'{mono_prefix}/mono.tok.{tgt_lang}', 'w') as f:
                    for i in range(line_counts):
                        f.write('{}\n'.format('。'))

    os.system(f"head {mono_prefix._str +'/'}mono.tok.{lang} -n 5")

def binarize(mono_prefix):
    binpath = Path('./DATA/data-bin', mono_dataset_name)
    src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
    tgt_dict_file = src_dict_file
    monopref = str(mono_prefix/"mono.tok") # whatever filepath you get after applying subword tokenization
    if binpath.exists():
        print(binpath, "exists, will not overwrite!")
    else:
        os.system(f"{config.python_path} -m fairseq_cli.preprocess \
            --source-lang 'zh'\
            --target-lang 'en'\
            --trainpref {monopref}\
            --destdir {binpath}\
            --srcdict {src_dict_file}\
            --tgtdict {tgt_dict_file}\
            --workers 2")

def generate_reverse_data():    
    # 將 binarized data 加入原本的資料夾中並用一個 split_name 取名
    # ex. ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]
    os.system(f"cp ./DATA/data-bin/mono/train.zh-en.zh.bin ./DATA/data-bin/ted2020/mono.zh-en.zh.bin")
    os.system(f"cp ./DATA/data-bin/mono/train.zh-en.zh.idx ./DATA/data-bin/ted2020/mono.zh-en.zh.idx")
    os.system(f"cp ./DATA/data-bin/mono/train.zh-en.en.bin ./DATA/data-bin/ted2020/mono.zh-en.en.bin")
    os.system(f"cp ./DATA/data-bin/mono/train.zh-en.en.idx ./DATA/data-bin/ted2020/mono.zh-en.en.idx")

    proj = "hw5.seq2seq"
    logger = logging.getLogger(proj)

    task = load_task(logger)

    model = build_model(arch_args, task)
    model = model.to(device=device)

    sequence_generator = task.build_generator([model], config)

    generate_prediction(model, task ,split="mono" ,outfile="./mono.en.txt", sequence_generator=sequence_generator)

def generate_dataset(mono_prefix):
    # 合併剛剛生成的 prediction_file (.en) 以及中文 mono.zh (.zh)
    # 
    # hint: 在此用剛剛的 spm model 對 prediction_file 進行切斷詞
    vocab_size = 8000

    data_dir = "/home/leyan/Documents/Hung-yi-Lee/hw6_transformer/DATA/rawdata/"
    prefix = Path(data_dir).absolute() / "ted2020"

    # spm_model.encode(line, out_type=str)
    # output: ./DATA/rawdata/mono/mono.tok.en & mono.tok.zh

    spm_model = spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))
    with open(f'{mono_prefix}/mono.finaltok.en', 'w') as out_f:
        with open('./mono.en.txt', 'r') as in_f:
            for line in in_f:
                line = line.strip()
                tok = spm_model.encode(line, out_type=str)
                print(' '.join(tok), file=out_f)

    os.system(f"cp {mono_prefix}/mono.tok.zh {mono_prefix}/mono.finaltok.zh")

    
    # hint: 在此用 fairseq 把這些檔案再 binarize
    binpath = Path('./DATA/data-bin/synthetic')
    src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
    tgt_dict_file = src_dict_file
    monopref = "./DATA/rawdata/mono/mono.tok" # or whatever path after applying subword tokenization, w/o the suffix (.zh/.en)
    if binpath.exists():
        print(binpath, "exists, will not overwrite!")
    else:
        os.system(f"{config.python_path} -m fairseq_cli.preprocess \
            --source-lang 'zh'\
            --target-lang 'en'\
            --trainpref {monopref}\
            --destdir {binpath}\
            --srcdict {src_dict_file}\
            --tgtdict {tgt_dict_file}\
            --workers 2")

    # 這裡用剛剛準備的檔案合併原先 ted2020 來生成最終 back-translation 的資料
    os.system(f"cp -r ./DATA/data-bin/ted2020/ ./DATA/data-bin/ted2020_with_mono/")

    os.system(f"cp ./DATA/data-bin/synthetic/train.zh-en.zh.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.bin")
    os.system(f"cp ./DATA/data-bin/synthetic/train.zh-en.zh.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.idx")
    os.system(f"cp ./DATA/data-bin/synthetic/train.zh-en.en.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.bin")
    os.system(f"cp ./DATA/data-bin/synthetic/train.zh-en.en.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.idx")
    
if __name__ == '__main__':
    mono_prefix = download_data()
    src_lang = 'zh'   
    tgt_lang = 'en' 

    # # file process  
    # clean_corpus(mono_prefix, src_lang)   

    data_prefix = f'{mono_prefix}/ted_zh_corpus'
    os.system(f"head {data_prefix+'.clean'} -n 5")

    # Subword Units
    subword_units(mono_prefix, src_lang, tgt_lang)

    # use fairseq transform to binary data
    binarize(mono_prefix)

    # alreay done
    # generate_reverse_data()

    generate_dataset(mono_prefix)