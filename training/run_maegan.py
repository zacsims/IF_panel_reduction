import os
import torch
import torch.nn as nn
import wandb
from skimage.io import imshow, imread
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import spearman_corrcoef as spearman
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mae import MAE
from vit_pytorch.vit import ViT
from einops import repeat, rearrange
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from pytorch_lightning.callbacks import ModelCheckpoint

class SingleCellDataset(Dataset):
    def __init__(self, files):
        self.img_files = files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filepath = self.img_files[idx]
        img = imread(filepath)
        tensor = torch.from_numpy(img)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        tensor /= 255.

        return tensor
    
    
class IF_MAEGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.mae =  MAE(encoder = ViT(image_size=160,
                                      patch_size=32,
                                      num_classes=1000,
                                      dim=1024,
                                      depth=6,
                                      heads=8,
                                      channels=1,
                                      mlp_dim=2048),
                        masking_ratio = 0.75,    # the paper recommended 75% masked patches
                        decoder_dim = 512,       # paper showed good results with just 512
                        decoder_depth = 6)       # anywhere from 1 to 8
        
        num_masked = int(self.mae.masking_ratio * 25)
        self.discriminator = nn.Sequential(ViT(image_size=32,
                                               patch_size=4,
                                               num_classes=1,
                                               dim=512,
                                               depth=4,
                                               heads=6,
                                               channels=num_masked,
                                               mlp_dim=1024),
                                           nn.Sigmoid())
        
    def forward(self, x, masked_patch_idx):
        real, fake = self.mae(x, masked_patch_idx=masked_patch_idx)
        return real, fake
    
    def configure_optimizers(self):
        mae_opt = torch.optim.Adam(self.mae.parameters(), lr=1e-3)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        return mae_opt, d_opt
  
    def training_step(self, train_batch, batch_idx):
        mae_opt, d_opt = self.optimizers()
        real, fake = self.mae(train_batch)
        
        ##############################
        # Optimize GAN Discriminator #
        ##############################
        real = rearrange(real, 'b c (h w) -> b c h w', h=32)
        fake = rearrange(fake, 'b c (h w) -> b c h w', h=32)
        
        real_label = torch.ones((train_batch.shape[0], 1), device=self.device)
        fake_label = torch.zeros((train_batch.shape[0], 1), device=self.device)

        r_loss = F.binary_cross_entropy(self.discriminator(real), real_label)
        d_opt.zero_grad()
        self.manual_backward(r_loss)
        f_loss = F.binary_cross_entropy(self.discriminator(fake.detach()), fake_label)
        self.manual_backward(f_loss)
        d_opt.step()
        
        ################
        # Optimize MAE #
        ################
        mae_gan_loss = F.binary_cross_entropy(self.discriminator(fake), real_label)
        mae_loss = F.mse_loss(fake, real)
        mae_opt.zero_grad()
        self.manual_backward(mae_loss + mae_gan_loss)
        mae_opt.step()
           
        self.log_dict({"mae_loss": mae_loss, "mae_gan_loss":mae_gan_loss,  "discriminator_loss": r_loss + f_loss}, sync_dist=True)
    
    def validation_step(self, val_batch, batch_idx):
        real, fake = self.mae(val_batch)
        #reshape to BxCxHxW
        real = rearrange(real, 'b c (h w) -> b c h w', h=32)
        fake = rearrange(fake, 'b c (h w) -> b c h w', h=32)
        #calculate mean intensities
        mean_int = torch.mean(real, dim=(2,3))
        pred_mean_int = torch.mean(fake, dim=(2,3))
        #calculate mean spearman correlation across reconstructed markers
        corr = torch.mean(spearman(pred_mean_int, mean_int))
        #get ssim
        ssim_score = ssim(real, fake)
        #log
        self.log('val_ssim', ssim_score, sync_dist=True)
        self.log('val_corr', corr, sync_dist=True)
        
        
if __name__ == '__main__':
    data_dir='/home/groups/ChangLab/simsz/panel_reduction/data/train/TMA4-CellTilesTiled'
    files = [f'{data_dir}/{f}' for f in os.listdir(data_dir)]
    
    split = round(len(files) * 0.9)
    train_files = files[:split]
    val_files = files [split:]
    
    train_data = SingleCellDataset(train_files)
    val_data = SingleCellDataset(val_files)
    
    BATCH_SIZE = 512
    train_loader = DataLoader(train_data, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=2,
                              persistent_workers=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=BATCH_SIZE,
                            num_workers=2,
                            persistent_workers=True,
                            pin_memory=True)
    
    wandb_logger = WandbLogger(project="pt_maegan", entity='changlab')
    model = IF_MAEGAN()
    checkpoint_callback = ModelCheckpoint(monitor="val_ssim", mode="max")
    trainer = pl.Trainer(accelerator='gpu',
                         devices=8,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback],
                         max_epochs=150,
                         num_sanity_val_steps=0,
                         strategy="ddp")
    trainer.fit(model, train_loader, val_loader)

