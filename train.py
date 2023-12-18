import torch
import wandb
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from piq import FID, ssim
from piq.feature_extractors import InceptionV3


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, gen_criterion, dis_criterion,
                 device, train_loader, val_loader, gen_lr_scheduler=None, dis_lr_scheduler=None, num_epoch=10):
        self.device = device

        self.generator = generator
        self.discriminator = discriminator
        self.gen_criterion = gen_criterion
        self.dis_criterion = dis_criterion
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.gen_lr_scheduler = gen_lr_scheduler
        self.dis_lr_scheduler = dis_lr_scheduler

        self.num_epochs = num_epoch
        self.train_data = train_loader
        self.val_data = val_loader

        self.fixed_noise = torch.randn(16, 100, 1, 1)
        self.inception = InceptionV3().to(self.device)
        self.inception.eval()
        self.fid = FID()

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        for i, batch in enumerate(tqdm(self.train_data, desc="Train epoch")):
            batch = batch.to(self.device)
            z = torch.randn(batch.shape[0], 100, 1, 1).to(self.device)
            generated = self.generator(z)

            self.dis_optimizer.zero_grad()

            dis_real = self.discriminator(batch)
            dis_gen = self.discriminator(generated.detach())

            dis_loss = self.dis_criterion(dis_real, dis_gen)

            dis_loss.backward()
            self.dis_optimizer.step()

            self.gen_optimizer.zero_grad()

            generated = self.generator(z)

            dis_real = self.discriminator(batch)
            dis_gen = self.discriminator(generated)

            gen_loss = self.gen_criterion(dis_gen)
            gen_loss.backward()
            self.gen_optimizer.step()

            gen_loss, dis_loss = gen_loss.item(), dis_loss.item()

            wandb.log(
                {
                    "generator_loss": gen_loss,
                    "discriminator_loss": dis_loss,
                    "SSIM": ssim(self._norm_image(batch.detach()), self._norm_image(generated.detach()))
                }
            )

    @torch.no_grad()
    def evaluate_epoch(self, epoch):
        self.generator.eval()
        gen_features = []
        real_features = []
        for i, batch in enumerate(tqdm(self.val_data, desc="Evaluate epoch")):
            z = torch.randn(batch.shape[0], 100, 1, 1).to(self.device)
            generated = self.generator(z)

            gen_features.append(self._feature_extractor(generated).cpu())
            real_features.append(self._feature_extractor(batch.to(self.device)).cpu())

        gen_features = torch.cat(gen_features, dim=0).to(self.device)
        real_features = torch.cat(real_features, dim=0).to(self.device)

        fid = self.fid(real_features, gen_features)

        wandb.log({"FID": fid.detach()})

    def log_prediction(self, epoch, rows=4, cols=4):
        self.generator.eval()

        z = torch.randn(rows * cols, 100, 1, 1).to(self.device)

        generated = self.generator(z)
        images = make_grid(generated.detach().cpu(), nrow=rows, normalize=True, value_range=(-1, 1))
        images = to_pil_image(images)
        wandb.log({"Generated samples": wandb.Image(images, caption="Generated samples")})

        images = self._norm_image(generated)
        images = to_pil_image(images[0].detach().cpu())
        wandb.log({"Generated sample": wandb.Image(images, caption="Generated sample")})

        generated = self.generator(self.fixed_noise.to(self.device))
        images = make_grid(generated.detach().cpu(), nrow=rows, normalize=True, value_range=(-1, 1))
        images = to_pil_image(images)
        wandb.log({"Generated fixed samples": wandb.Image(images, caption="Generated fixed samples")})

    def save_checkpoint(self, epoch):
        arch = type(self.generator).__name__

        state = {
            "arch": arch,
            "state_dict": self.generator.state_dict(),
            "optimizer": self.gen_optimizer.state_dict()
        }
        torch.save(state, f"checkpoint_{epoch + 1}.pth")
        print("Saving checkpoint...")

    def _norm_image(self, image):
        return (image + 1) / 2

    def _feature_extractor(self, images):
        images = self._norm_image(images)
        features = self.inception(images)
        return features[0].flatten(1)

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            if epoch % 5 == 0:
                self.evaluate_epoch(epoch)
            self.log_prediction(epoch)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

