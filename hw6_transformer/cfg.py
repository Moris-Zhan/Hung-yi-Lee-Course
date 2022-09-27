from argparse import Namespace

# For baseline, medium
# config = Namespace(
#     python_path = "/home/leyan/anaconda3/envs/torch_1_21/bin/python",
#     datadir = "./DATA/data-bin/ted2020",
#     # savedir = "./checkpoints/rnn", # baseline
#     savedir = "./checkpoints/transformer",  # medium
#     source_lang = "en",
#     target_lang = "zh",
    
#     # cpu threads when fetching & processing data.
#     num_workers=2,  
#     # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
#     max_tokens=8192,
#     accum_steps=2,
    
#     # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
#     lr_factor=2.,
#     lr_warmup=4000,
    
#     # clipping gradient norm helps alleviate gradient exploding
#     clip_norm=1.0,
    
#     # maximum epochs for training
#     max_epoch=30,
#     start_epoch=1,
    
#     # beam size for beam search
#     beam=5, 
#     # generate sequences of maximum length ax + b, where x is the source length
#     max_len_a=1.2, 
#     max_len_b=10,
#     # when decoding, post process sentence by removing sentencepiece symbols.
#     post_process = "sentencepiece",
    
#     # checkpoints
#     keep_last_epochs=5,
#     resume=None, # if resume from checkpoint name (under config.savedir)
    
#     # logging
#     use_wandb=False,
# )

# For Strong(Back-translation)
config = Namespace(
    python_path = "/home/leyan/anaconda3/envs/torch_1_21/bin/python",
    datadir = "./DATA/data-bin/ted2020_with_mono",
    savedir = "./checkpoints/rnn-bt",  # strong
    source_lang = "en",
    target_lang = "zh",
    
    # cpu threads when fetching & processing data.
    num_workers=2,  
    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,
    
    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor=2.,
    lr_warmup=4000,
    
    # clipping gradient norm helps alleviate gradient exploding
    clip_norm=1.0,
    
    # maximum epochs for training
    max_epoch=30,
    start_epoch=1,
    
    # beam size for beam search
    beam=5, 
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a=1.2, 
    max_len_b=10,
    # when decoding, post process sentence by removing sentencepiece symbols.
    post_process = "sentencepiece",
    
    # checkpoints
    keep_last_epochs=5,
    resume=None, # if resume from checkpoint name (under config.savedir)
    
    # logging
    use_wandb=False,
)