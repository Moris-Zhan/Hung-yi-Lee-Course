import re
from pathlib import Path
import os
import random
import sentencepiece as spm
# fairseq_cli
from fairseq import utils
import subprocess
from cfg import config

data_dir = './DATA/rawdata'
dataset_name = 'ted2020'

def download_data():     
    urls = (
        # '"https://onedrive.live.com/download?cid=3E549F3B24B238B4&resid=3E549F3B24B238B4%214989&authkey=AGgQ-DaR8eFSl1A"', 
        # '"https://onedrive.live.com/download?cid=3E549F3B24B238B4&resid=3E549F3B24B238B4%214987&authkey=AA4qP_azsicwZZM"',
    # # If the above links die, use the following instead. 
        "https://www.csie.ntu.edu.tw/~r09922057/ML2021-hw5/ted2020.tgz",
        "https://www.csie.ntu.edu.tw/~r09922057/ML2021-hw5/test.tgz",
    # # If the above links die, use the following instead. 
    #     "https://mega.nz/#!vEcTCISJ!3Rw0eHTZWPpdHBTbQEqBDikDEdFPr7fI8WxaXK9yZ9U",
    #     "https://mega.nz/#!zNcnGIoJ!oPJX9AvVVs11jc0SaK6vxP_lFUNTkEcK2WbxJpvjU5Y",
    )
    file_names = (
        'ted2020.tgz', # train & dev
        'test.tgz', # test
    )
    prefix = Path(data_dir).absolute() / dataset_name
    prefix.mkdir(parents=True, exist_ok=True)

    for u, f in zip(urls, file_names):
        path = prefix/f
        if not path.exists():
            if 'mega' in u:
                os.system(f"megadl {u} --path {path}")
            else:
                os.system(f"wget {u} -O {path}")
        if path.suffix == ".tgz":
            os.system(f"tar -xvf {path} -C {prefix}")
        elif path.suffix == ".zip":
            os.system(f"unzip -o {path} -d {prefix}")
    
    os.system(f"mv {prefix/'raw.en'} {prefix/'train_dev.raw.en'}")
    os.system(f"mv {prefix/'raw.zh'} {prefix/'train_dev.raw.zh'}")
    os.system(f"mv {prefix/'test.en'} {prefix/'test.raw.en'}")
    os.system(f"mv {prefix/'test.zh'} {prefix/'test.raw.zh'}")
    return prefix

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

def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return
    with open(f'{prefix}.{l1}', 'r') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0: # remove short sentence
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0: # remove long sentence
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0: # remove by ratio of length
                            if s1_len/s2_len > ratio or s2_len/s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)

def split_data():
    valid_ratio = 0.01 # 3000~4000句就夠了
    train_ratio = 1 - valid_ratio

    if (prefix/f'train.clean.{src_lang}').exists() \
    and (prefix/f'train.clean.{tgt_lang}').exists() \
    and (prefix/f'valid.clean.{src_lang}').exists() \
    and (prefix/f'valid.clean.{tgt_lang}').exists():
        print(f'train/valid splits exists. skipping split.')
    else:
        line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}'))
        labels = list(range(line_num))
        random.shuffle(labels)
        for lang in [src_lang, tgt_lang]:
            train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w')
            valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w')
            count = 0
            for line in open(f'{data_prefix}.clean.{lang}', 'r'):
                if labels[count]/line_num < train_ratio:
                    train_f.write(line)
                else:
                    valid_f.write(line)
                count += 1
            train_f.close()
            valid_f.close()

def subword_units():
    print(prefix)
    vocab_size = 8000
    if (prefix/f'spm{vocab_size}.model').exists():
        print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
    else:
        spm.SentencePieceTrainer.train(
            input=','.join([f'{prefix}/train.clean.{src_lang}',
                            f'{prefix}/valid.clean.{src_lang}',
                            f'{prefix}/train.clean.{tgt_lang}',
                            f'{prefix}/valid.clean.{tgt_lang}']),
            model_prefix=prefix/f'spm{vocab_size}',
            vocab_size=vocab_size,
            character_coverage=1,
            model_type='unigram', # 'bpe' 也可
            input_sentence_size=1e6,
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc_cf',
        )

    spm_model = spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))
    in_tag = {
        'train': 'train.clean',
        'valid': 'valid.clean',
        'test': 'test.raw.clean',
    }

    # already process   
    for split in ['train', 'valid', 'test']:
        for lang in [src_lang, tgt_lang]:
            out_path = prefix/f'{split}.{lang}'
            if out_path.exists():
                print(f"{out_path} exists. skipping spm_encode.")
            else:
                with open(prefix/f'{split}.{lang}', 'w') as out_f:
                    with open(prefix/f'{in_tag[split]}.{lang}', 'r') as in_f:
                        for line in in_f:
                            line = line.strip()
                            tok = spm_model.encode(line, out_type=str)
                            print(' '.join(tok), file=out_f)

    os.system(f"head {data_dir+'/'+dataset_name+'/train.'+src_lang} -n 5")
    os.system(f"head {data_dir+'/'+dataset_name+'/train.'+tgt_lang} -n 5")

def failseq_to_binary_data():
    binpath = Path('./DATA/data-bin', dataset_name)
    if binpath.exists():
        print(binpath, "exists, will not overwrite!")
    else:
        python_path = subprocess.check_output("which python", shell=True)
        python_path = str(python_path).strip()
        # python_path = "/home/leyan/anaconda3/envs/torch_1_21/bin/python"
        # print(f"{prefix/'train'}")   # /home/leyan/Documents/Hung-yi-Lee/hw6_transformer/DATA/rawdata/ted2020/train
        os.system(f"{config.python_path} -m fairseq_cli.preprocess \
            --source-lang {src_lang}\
            --target-lang {tgt_lang}\
            --trainpref {prefix/'train'}\
            --validpref {prefix/'valid'}\
            --testpref {prefix/'test'}\
            --destdir {binpath}\
            --joined-dictionary\
            --workers 2")

if __name__ == '__main__':
    prefix = download_data()
    src_lang = 'en'
    tgt_lang = 'zh'

    data_prefix = f'{prefix}/train_dev.raw'
    test_prefix = f'{prefix}/test.raw'

    # os.system(f"head {data_prefix+'.'+src_lang} -n 5")
    # os.system(f"head {data_prefix+'.'+tgt_lang} -n 5")

    # # file process   
    # clean_corpus(data_prefix, src_lang, tgt_lang)
    # clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)     

    # os.system(f"head {data_prefix+'.clean.'+src_lang} -n 5")
    # os.system(f"head {data_prefix+'.clean.'+tgt_lang} -n 5")

    # split train/valid set
    split_data()

    # Subword Units
    subword_units()

    # use fairseq transform to binary data
    failseq_to_binary_data()
    