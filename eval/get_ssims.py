import torch
import gc
from tqdm import tqdm
from einops import rearrange
from torchmetrics.functional import structural_similarity_index_measure as ssim

def get_ssims(model, panel, val_loader, device='cpu'):
    masked_ch_idx = torch.tensor([i for i in range(25) if i not in panel], device=device)
    model.mae.masking_ratio = (25-len(panel))/25

    with torch.no_grad():
        ssims= torch.empty(len(val_loader.dataset), device=device)

        for i, (batch, fnames) in enumerate(tqdm(val_loader)):
            batch = batch.to(device)
            masked,preds = model.forward(batch, masked_patch_idx=masked_ch_idx)
            preds = rearrange(preds, 'b c (h w) -> b c h w', h=32)
            masked = rearrange(masked, 'b c (h w) -> b c h w', h=32)
            ssims_ = ssim(masked, preds)
            s = i * len(batch)
            e = s + len(batch)
            ssims[s:e] = ssims_

            del batch
            if device != 'cpu':
                gc.collect()
                torch.cuda.empty_cache()
        
    return ssims