import argparse

import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from model import Generator


RESUME_PATH = 'model.pth'

IN_CHANNELS = [100, 512, 256, 128, 64]
OUT_CHANNELS = [512, 256, 128, 64, 3]
KERNEL_SIZE = [4, 4, 4, 4, 4]
STRIDE = [1, 2, 2, 2, 2]
PADDING = [0, 1, 1, 1, 1]


def main(n):
    assert n > 0, "number of samples must be greater than 0"

    print("Preparing model")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator = Generator(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING).to(device)

    checkpoint = torch.load(RESUME_PATH, device)
    generator.load_state_dict(checkpoint["state_dict"])

    print(generator)

    generator.eval()
    z = torch.randn(n, 100, 1, 1, device=device)
    images = generator(z)
    images = make_grid(images.cpu(), nrow=min(n, 5), normalize=True, value_range=(-1, 1))
    images = to_pil_image(images)

    plt.imshow(images)
    plt.savefig("generated_samples")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-n",
        "--sample-number",
        default=1,
        type=int,
        help="Number of samples to generate",
    )
    args = args.parse_args()
    main(args.sample_number)

