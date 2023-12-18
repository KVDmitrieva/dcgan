import torch
import argparse
import wandb

from torch.utils.data import DataLoader

from model import Generator, Discriminator
from dataset import AnimeDataset
from loss import DiscriminatorLoss, GeneratorLoss
from train import Trainer


BATCH_SIZE = 512

IN_CHANNELS = [100, 512, 256, 128, 64]
OUT_CHANNELS = [512, 256, 128, 64, 3]
KERNEL_SIZE = [4, 4, 4, 4, 4]
STRIDE = [1, 2, 2, 2, 2]
PADDING = [0, 1, 1, 1, 1]

LR = 2e-4
BETAS = (0.5, 0.999)

NUM_EPOCHS = 100

DATA_PATH = "data/images"


def main(wandb_key=None):
    if wandb_key is not None:
        wandb.login(key=wandb_key)
    wandb.init(project="gan")

    train_data = AnimeDataset(DATA_PATH, train=True)
    val_data = AnimeDataset(DATA_PATH, train=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    generator = Generator(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING).to(device)
    discriminator = Discriminator(OUT_CHANNELS[::-1], IN_CHANNELS[::-1], KERNEL_SIZE[::-1], STRIDE[::-1],
                                  PADDING[::-1]).to(device)

    print(generator)
    print(discriminator)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)

    gen_criterion = GeneratorLoss()
    dis_criterion = DiscriminatorLoss()

    trainer = Trainer(generator, discriminator, gen_optimizer, dis_optimizer, gen_criterion, dis_criterion,
                      device, train_loader, val_loader, num_epoch=NUM_EPOCHS)

    trainer.train()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-k",
        "--key",
        default=None,
        type=str,
        help="wandb key for logging",
    )
    args = args.parse_args()
    main(args.key)
