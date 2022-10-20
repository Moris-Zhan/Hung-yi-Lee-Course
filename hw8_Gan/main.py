# import module

# baseline
# from models.dcgan.trainer import TrainerGAN, get_config

# medium
# from models.wgan.trainer import TrainerGAN, get_config

# strong
from models.wgan_gp.trainer import TrainerGAN, get_config

from utils import same_seeds

same_seeds(2022)
workspace_dir = '.'


if __name__ == '__main__':
    trainer = TrainerGAN(get_config(workspace_dir))
    # trainer.train()

    # save the 1000 images into ./output folder
    trainer.inference(f'{workspace_dir}/checkpoints/2022-10-19_18-59-13_WGAN_GP/G_19.pth') # you have to modify the path when running this line