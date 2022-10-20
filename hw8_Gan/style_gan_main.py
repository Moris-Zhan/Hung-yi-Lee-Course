# import module
from datetime import datetime

import os
from retry.api import retry_call
from tqdm import tqdm

from models.style_gan.util_func import NanException


# boss
from models.style_gan.trainer import TrainerGAN 

from utils import same_seeds, timestamped_filename

same_seeds(2022)
workspace_dir = '.'

import os

config = {
        "model_type": "STYLE_GAN",
        "batch_size": 32,
        "lr": 1e-4,
        "n_epoch": 20,
        "n_critic": 1,
        "z_dim": 100,
        "clip_value": 1.0,
        "workspace_dir": workspace_dir, # define in the environment setting
    }
# update dir by time
time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# results_dir = os.path.join(workspace_dir, time+f'_{config["model_type"]}', "results")
# models_dir = os.path.join(workspace_dir, time+f'_{config["model_type"]}', "models")

results_dir = os.path.join(workspace_dir+"/logs", time+f'_{config["model_type"]}')
models_dir = os.path.join(workspace_dir+"/checkpoints", time+f'_{config["model_type"]}')



if __name__ == '__main__':
    model_args = dict(
        data = os.path.join(config["workspace_dir"], 'faces'),                        
        results_dir = results_dir,
        models_dir = models_dir,
        name = 'stylegan2',      
        image_size = 64,
        batch_size = 32,      
        n_epoch = 20,
        learning_rate = 2e-4,
        lr_mlp = 0.1,        
        num_workers =  32,      
        generate = True,
        interpolation_num_steps = 100,
        save_frames = True,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
    )
    trainer = TrainerGAN(**model_args)
    trainer.load(num=-1)
    trainer.set_data_src(model_args["data"])

    # trainer.num_train_steps = 10

    for epoch in range(model_args["n_epoch"]):
        progress_bar = tqdm(initial = 0, total = trainer.num_train_steps, mininterval=10., 
            desc=f'{model_args["name"]} | Epoch : [{epoch} / {model_args["n_epoch"]}]')

        for i in range(trainer.num_train_steps):
            retry_call(trainer.train, tries=3, exceptions=NanException)
            progress_bar.n = trainer.steps -(epoch * trainer.num_train_steps)
            progress_bar.refresh()
            # if trainer.steps % 50 == 0:
            #     trainer.save(trainer.checkpoint_num)    
        trainer.print_log()          
        trainer.save(epoch)    


        samples_name = timestamped_filename()
        trainer.generate_interpolation(samples_name, num_steps = model_args["interpolation_num_steps"])
        print(f'interpolation generated at {results_dir}/{model_args["name"]}/{samples_name}')

        # save the 1000 images into ./output folder
        trainer.inference() # you have to modify the path when running this line

    # save the 1000 images into ./output folder
    trainer.inference(f'{workspace_dir}/checkpoints/2022-10-20_15-58-04_STYLE_GAN/stylegan2/model_3.pt') # you have to modify the path when running this line
