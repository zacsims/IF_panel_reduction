import os
import sys
import pickle
import torch
import wandb
import numpy as np
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



class SingleCellDataset(Dataset):
    def __init__(self, files):
        self.img_files = files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filepath = self.img_files[idx]
        im = imread(filepath)
        num_channels = im.shape[-1]
        step_size = int(np.sqrt(num_channels))
        im = np.concatenate([np.concatenate([im[:,:,ch] for ch in range(i, i+step_size)], axis=1)  for i in range(0,num_channels-step_size+1,step_size)], axis=0)
        tensor = torch.from_numpy(im)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()

        return tensor
    
    
class IF_MAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.mae =  MAE(encoder = ViT(image_size=160, #128, 160
                                      patch_size=32,
                                      num_classes=1000,
                                      dim=1024,
                                      depth=6,
                                      heads=8,
                                      channels=1,
                                      mlp_dim=2048),
                        masking_ratio = 0.5,    # the paper recommended 75% masked patches
                        decoder_dim = 512,       # paper showed good results with just 512
                        decoder_depth = 12)       # anywhere from 1 to 8
        
    def forward(self, x, masked_patch_idx):
        masked_patches, pred_pixel_values = self.mae(x, masked_patch_idx=masked_patch_idx)
        return pred_pixel_values
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        masked_patches, pred_pixel_values = self.mae(train_batch)
        loss = F.mse_loss(pred_pixel_values, masked_patches)
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        real, fake = self.mae(val_batch)
        loss = F.mse_loss(fake, real)
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
        self.log('val_loss', loss, sync_dist=True)
        
        
if __name__ == '__main__':
    #data_dir='/home/groups/ChangLab/simsz/panel_reduction/data/train/TMA4-CellTilesTiled'
    data_dir = '/var/local/ChangLab/panel_reduction/CRC-TMA-2'
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
                              num_workers=8,
                              persistent_workers=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=BATCH_SIZE,
                            num_workers=0)
                            #persistent_workers=True,
                            #pin_memory=True)
                            
    
    wandb_logger = WandbLogger(project="pt_mae", entity='changlab', resume='allow')
    pl.seed_everything(69, workers=True)
    torch.set_float32_matmul_precision('high')
    model = IF_MAE()
    trainer = pl.Trainer(accelerator='gpu',
                         devices=8,
                         logger=wandb_logger,
                         max_epochs=300,
                         num_sanity_val_steps=0,
                         deterministic=True,
                         strategy='ddp')
    trainer.fit(model, train_loader, val_loader)

    
    
    
    
