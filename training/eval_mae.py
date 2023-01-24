import os
import torch
import wandb
from skimage.io import imshow, imread
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import spearman_corrcoef as spearman
from torch.utils.data import DataLoader
import torch.nn.functional as F
from attention_guided_mae import MAE, ViT
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
        img = imread(filepath)
        tensor = torch.from_numpy(img)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        #tensor /= 255.

        return tensor
    
    
class IF_MAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
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
        self.save_hyperparameters()
        
    def forward(self, x, masked_patch_idx=None, mask_with_attention=False, return_attention=False, mask_after_attention=False):
        if return_attention or mask_with_attention:
            masked_patches, pred_pixel_values, attn_map= self.mae(x, 
                                                                  masked_patch_idx=masked_patch_idx,
                                                                  mask_with_attention=mask_with_attention,
                                                                  return_attention=return_attention,
                                                                  mask_after_attention=mask_after_attention)
            return masked_patches, pred_pixel_values, attn_map
        
        masked_patches, pred_pixel_values = self.mae(x, 
                                                     masked_patch_idx=masked_patch_idx,
                                                     mask_after_attention=mask_after_attention)
        return masked_patches, pred_pixel_values
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        masked_patches, pred_pixel_values = self.mae(train_batch)
        loss = F.mse_loss(pred_pixel_values, masked_patches)
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        orig, preds = self.mae(val_batch)
        loss = F.mse_loss(preds, orig)
        #reshape to BxCxHxW
        preds = rearrange(preds, 'b c (h w) -> b c h w', h=32)
        orig = rearrange(orig, 'b c (h w) -> b c h w', h=32)
        #calculate mean intensities
        mean_int = torch.mean(orig, dim=(2,3))
        pred_mean_int = torch.mean(preds, dim=(2,3))
        #calculate mean spearman correlation across reconstructed markers
        corr = torch.mean(spearman(pred_mean_int, mean_int))
        #get ssim
        ssim_score = ssim(orig, preds)
        #log
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_ssim', ssim_score, sync_dist=True)
        self.log('val_corr', corr, sync_dist=True)
