import torch
import gc
from skimage.io import imread
from tqdm import tqdm
from einops import rearrange, repeat
from torchmetrics.functional import spearman_corrcoef as spearman

def get_mints(unmasked, gt, preds, fnames, device):
    #reshape to BxCxHxW
    preds = rearrange(preds, 'b c (h w) -> b c h w', h=32)
    gt = rearrange(gt, 'b c (h w) -> b c h w', h=32)
    unmasked = rearrange(unmasked.squeeze(1), 'b (c1 h) (c2 w) -> b (c1 c2) h w',h=32, w=32)
    preds[preds < 0] = 0
    
    if 'TMA004' in fnames[0]:
        #reconstruct cell nucleus mask by binarizing dapi channel
        mask = unmasked[:,0].clone()
    else:
        #load cell masks
        mask = torch.zeros((gt.shape[0], 32, 32), device=device)
        for i,fname in enumerate(fnames):
            fname = fname.replace('.tif', '-mask.tif')
            dir_ = '/var/local/ChangLab/panel_reduction/CRC-WSI-cell-masks'
            if 'TMA' in fname:
                dir_ = '/var/local/ChangLab/panel_reduction/CRC-TMA-2-cell-masks'           
            mask[i] = torch.tensor(imread(f'{dir_}/{fname}'), device=device)[:,:,0]
        
    #set mask to bool
    mask = mask.bool()
    
    #expand dapi mask to number of reconstructed channels
    mask = repeat(mask,'b h w -> b c h w', c=preds.shape[1])

    #calculate mean intensities
    mints = (gt * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))
    pmints = (preds * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))

    return mints, pmints


def get_intensities(model, panel, val_loader, device='cpu'):
    max_panel_size = 25
    masked_ch_idx = torch.tensor([i for i in range(max_panel_size) if i not in panel], device=device)
    model.mae.masking_ratio = (max_panel_size - len(panel)) / max_panel_size

    with torch.no_grad():
        #set tensors on cpu to store predicted and ground-truth mean intensities 
        mints= torch.zeros((len(val_loader.dataset), len(masked_ch_idx)), device='cpu')
        pmints = mints.clone()

        #run each batch through the model and copy intensities back to cpu
        for i, (batch, fnames) in enumerate(tqdm(val_loader)):
            batch = batch.to(device)
            gt,preds = model.forward(batch, masked_patch_idx=masked_ch_idx)
            mints_, pmints_ = get_mints(batch, gt, preds, fnames, device)
            s = i * len(batch)
            e = s + len(batch)
            mints_ = mints_.to('cpu')
            pmints_ = pmints_.to('cpu')
            mints[s:e, :] = mints_
            pmints[s:e, :] = pmints_

            del batch
            if device != 'cpu':
                gc.collect()
                torch.cuda.empty_cache()
                     
    return mints, pmints
