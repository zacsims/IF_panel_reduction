print('loading libraries')
import torch
from eval_mae import IF_MAE
from ssim_spearman_eval import ssim_corr_eval
import pytorch_lightning as pl

print('loading data')
val_data = torch.load('../data/TMA4_val.pt')
val_data = val_data
print('loading models')
model_50p_cls = IF_MAE().load_from_checkpoint('/home/users/simsz/changlab/simsz/panel_reduction/training/pt_mae/3f03m9po/checkpoints/epoch=114-step=17595.ckpt')
model_50p = IF_MAE().load_from_checkpoint('/home/users/simsz/changlab/simsz/panel_reduction/training/pt_mae/2zg7ig4z/checkpoints/epoch=38-step=5967.ckpt')
model_75p = IF_MAE().load_from_checkpoint('/home/users/simsz/changlab/simsz/panel_reduction/training/pt_mae/1orje2q0/checkpoints/epoch=99-step=15300.ckpt')
model_75p_cls = IF_MAE().load_from_checkpoint('/home/users/simsz/changlab/simsz/panel_reduction/training/pt_mae/1gu0c9tr/checkpoints/epoch=84-step=13005.ckpt')
model_50p.mae.masking_ratio = .50
model_50p_cls.mae.masking_ratio = .28
model_75p_cls.mae.masking_ratio = .28
models = [('50p masking', model_50p),('50p masking with CLS token', model_50p_cls), ('75p masking', model_75p), ('75p masking with CLS token', model_75p_cls)]

#trainer = pl.Trainer(accelerator='gpu', devices=8)

for model_name, model in models:
    if 'CLS' not in model_name: continue
    print()
    print(f'evaluating {model_name}')
    #random masking
    with torch.no_grad():
        preds, masked = model.forward(val_data)
    ssim_score, corr, _  = ssim_corr_eval(val_data, masked, preds)
    print(f'random masking strategy: SSIM:{ssim_score} Spearman:{corr}')
   
    if 'CLS token' in model_name: 
        with torch.no_grad():
            preds, masked, _ = model.forward(val_data,return_attention=True, mask_after_attention=True, mask_with_attention=True) 
        ssim_score, corr, _  = ssim_corr_eval(val_data, masked, preds)
        print(f'attention masking strategy: SSIM:{ssim_score} Spearman:{corr}')
   
    #correlation based panels
    #panel_channels = [0,22,7] #3 markers
    #panel_channels = [0,4,5,8,22,23] #6 markers
    #panel_channels = [0,4,5,8,10,12,18,22,23] 39 markers
    #panel_channels = [0,3,4,5,10,12,14,15,18,22,23,24] #12 markers
    #panel_channels = [0,1,2,3,4,5,6,10,12,14,15,18,22,23,24] #15 markers
    panel_channels = [0,1,2,3,4,5,6,8,10,11,12,14,15,18,20,22,23,24] #18 markers
    #if '75p masking' in model_name: panel_channels = [24, 17, 14, 5, 3, 18, 9]#panel_channels = [0,4,17,16,18,13,10]
    ch2stain = {0:"DAPI", 1:"CD3", 2:"ERK-1", 3:"hRAD51", 4:"CyclinD1", 5:"VIM", 6:"aSMA", 7:"ECad", 8:"ER", 9:"PR",
            10:"EGFR", 11:"Rb", 12:"HER2", 13:"Ki67", 14:"CD45", 15:"p21", 16:"CK14", 17:"CK19", 18:"CK17",
            19:"LaminABC", 20:"Androgen Receptor", 21:"Histone H2AX", 22:"PCNA", 23:"PanCK", 24:"CD31"}
    print(f'unmasked markers:{[ch2stain[ch] for ch in panel_channels]}')
    masked_ch_idx = torch.tensor([i for i in range(25) if i not in panel_channels])

    with torch.no_grad():
        preds, masked = model.forward(val_data,  masked_patch_idx=masked_ch_idx)
    ssim_score, corr, _  = ssim_corr_eval(val_data, masked, preds)
    print(f'chosen  masking strategy: SSIM:{ssim_score} Spearman:{corr}') 
