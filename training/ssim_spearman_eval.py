import torch
from einops import rearrange, repeat
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import spearman_corrcoef as spearman

def ssim_corr_eval(ims, masked, preds):
    #reshape to BxCxHxW
    preds = rearrange(preds, 'b c (h w) -> b c h w', h=32)
    masked_unpatched = rearrange(masked, 'b c (h w) -> b c h w', h=32)
    ims_unpatched = rearrange(ims.squeeze(1), 'b (c1 h) (c2 w) -> b (c1 c2) h w',h=32, w=32)

    #reconstruct cell nucleus mask by binarizing dapi channel
    dapi_mask = ims_unpatched[:,0].clone()
    dapi_mask[dapi_mask > 0] = 1
    #set zero values to nan so marker intensity inside cell boundary can easily be calculated with nanmean
    dapi_mask[dapi_mask == 0] = float('nan')

    #expand dapi mask to number of reconstructed channels
    dapi_mask = repeat(dapi_mask,'b h w -> b c h w', c=preds.shape[1])

    #"segment" cell images 
    segged_channels = masked_unpatched * dapi_mask
    pred_segged_channels = preds * dapi_mask

    #reconstructed image has values less than zero
    pred_segged_channels[pred_segged_channels < 0] = 0

    #calculate mean intensities
    mean_int = torch.nanmean(segged_channels, dim=(2,3))
    pred_mean_int = torch.nanmean(pred_segged_channels, dim=(2,3))

    #calculate mean spearman correlation across reconstructed markers
    corr = torch.mean(spearman(pred_mean_int, mean_int))

    #set nan values back to zero to calculate SSIM
    pred_segged_channels[torch.isnan(pred_segged_channels).bool()] = 0
    
    #get ssim
    ssim_score = ssim(masked_unpatched, pred_segged_channels)
    
    return ssim_score, corr, (mean_int, pred_mean_int)
