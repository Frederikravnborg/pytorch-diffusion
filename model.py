import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from modules import *
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from pytorch_fid.inception import InceptionV3
from scipy.stats import entropy
import torch.nn.functional as F
import wandb


class DiffusionModel(pl.LightningModule):
    def __init__(self, in_size, t_range, img_depth):
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size

        bilinear = True
        self.inc = DoubleConv(img_depth, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, img_depth)
        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(256, 4)
        self.sa3 = SAWrapper(128, 8)

        # Use the pool3 layer of InceptionV3 for feature extraction
        self.inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(self.device)
        self.inception_model.eval()
        self.generated_images = []


    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 16)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 8)
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, 16)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, 32)
        output = self.outc(x)
        return output

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small
        )

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss


    def denoise_sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            for i in range(t, 0, -1):
                e_hat = self.forward(x, torch.tensor([i], device=x.device, dtype=torch.float32).view(1, 1).repeat(x.shape[0], 1))
                if i > 1:
                    z = torch.randn(x.shape, device=x.device)
                else:
                    z = torch.zeros(x.shape, device=x.device)
                pre_scale = 1 / math.sqrt(self.alpha(i))
                e_scale = (1 - self.alpha(i)) / math.sqrt(1 - self.alpha_bar(i))
                post_sigma = math.sqrt(self.beta(i)) * z
                x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x


    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val_loss", loss)

        # Generate and log images
        with torch.no_grad():
            generated_images = self.denoise_sample(batch.to(self.device), torch.tensor([self.t_range - 1], device=self.device))
            self.generated_images.append(generated_images)

        return loss
    
    def on_validation_epoch_end(self):
        # Generate noise
        noise = torch.randn((32, 3, 32, 32), device=self.device)  # Adjust dimensions according to your dataset
        generated_images = self.denoise_sample(noise, self.t_range)
        
        # Log generated images
        self.log_images(generated_images, self.current_epoch)
        
        # Calculate and log Inception Score
        inception_score, inception_std = self.calculate_inception_score(generated_images)
        wandb.log({'inception_score': inception_score, 'inception_score_std': inception_std})


    def calculate_inception_score(self, images, splits=10):
        N = len(images)
        assert N > 0
        dataloader = DataLoader(images, batch_size=32)

        preds = np.zeros((N, 2048))  # Changed to match the feature dimension

        for i, batch in enumerate(dataloader, 0):
            batch = batch.to(self.device)
            with torch.no_grad():
                pred = self.inception_model(batch)[0]
            pred = pred.squeeze(-1).squeeze(-1)  # Squeeze to remove dimensions (1, 1)
            preds[i * 32: i * 32 + pred.size(0)] = pred.cpu().numpy()

        # Calculate Inception Score
        split_scores = []

        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)


    def log_images(self, images, epoch_idx, prefix="generated"):
        images = (images - images.min()) / (images.max() - images.min())
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        
        # Select indices for images to log: first, middle, and last
        indices_to_log = [0, len(images) // 2, len(images) - 1]
        
        for i in indices_to_log:
            img = images[i]
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            self.logger.experiment.log({f"{prefix}/_image_{i}": [wandb.Image(plt, caption=f"{prefix}_image_{i}")]})
            plt.close()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


