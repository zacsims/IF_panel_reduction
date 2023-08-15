import torch
import gc
from tqdm import tqdm
from einops import rearrange
from torchmetrics.functional import structural_similarity_index_measure as ssim

def get_ssims(model, panel, val_loader, device='cpu'):
    masked_ch_idx = torch.tensor([i for i in range(25) if i not in panel], device=device)
    model.mae.masking_ratio = (25-len(panel))/25

    with torch.no_grad():
        ssims= torch.zeros(len(val_loader.dataset), device='cpu')
        batch_size = None
        for i, (batch, _) in enumerate(tqdm(val_loader)):
            batch = batch.to(device)
            if batch_size is None: batch_size = len(batch)
            masked,preds = model.forward(batch, masked_patch_idx=masked_ch_idx)
            preds = rearrange(preds, 'b c (h w) -> b c h w', h=32)
            preds[preds < 0] = 0
            masked = rearrange(masked, 'b c (h w) -> b c h w', h=32)
            ssims_ = ssim(masked, preds, reduction='none')
            s = i * batch_size
            e = s + batch_size
            ssims_ = ssims_.to('cpu')
            ssims[s:e] = ssims_

            del batch
            if device != 'cpu':
                gc.collect()
                torch.cuda.empty_cache()
        
    return ssims