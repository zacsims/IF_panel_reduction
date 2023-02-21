print('loading libraries')
import torch
from  mae import IF_MAE
from ssim_spearman_eval import ssim_corr_eval

print('loading data')
val_data = torch.load('../data/TMA4_val.pt')
cut = len(val_data)
val_data = val_data[:cut]
print('loading models')


#88p enc dec cls
#model = IF_MAE().load_from_checkpoint('/home/users/simsz/changlab/simsz/panel_reduction/training/pt_mae/covlf9bw/checkpoints/epoch=119-step=18360.ckpt')
#50p cls
#model = IF_MAE().load_from_checkpoint('/home/users/simsz/changlab/simsz/panel_reduction/training/pt_mae/3f03m9po/checkpoints/epoch=114-step=17595.ckpt')
model = IF_MAE().load_from_checkpoint('/home/users/simsz/changlab/simsz/panel_reduction/training/pt_mae/1gu0c9tr/checkpoints/epoch=84-step=13005.ckpt')
log = 'log_75p.txt'
ch2stain = {0:"DAPI", 1:"CD3", 2:"ERK-1", 3:"hRAD51", 4:"CyclinD1", 5:"VIM", 6:"aSMA", 7:"ECad", 8:"ER", 9:"PR",
            10:"EGFR", 11:"Rb", 12:"HER2", 13:"Ki67", 14:"CD45", 15:"p21", 16:"CK14", 17:"CK19", 18:"CK17",
            19:"LaminABC", 20:"Androgen Receptor", 21:"Histone H2AX", 22:"PCNA", 23:"PanCK", 24:"CD31"}

top_panel = [0]
for num_masked in reversed(range(1,24)):
    masking_ratio = num_masked/25
    model.mae.masking_ratio = masking_ratio
    with open(log, 'a') as f:
        f.write(f'***best {25 - num_masked}  panel ***\n')
    top_corr = -999
    top_ch = None
    for ch_candidate in range(25):
        if ch_candidate in top_panel: continue
        if ch_candidate == 0: continue
        candidate_panel = top_panel.copy()
        candidate_panel.append(ch_candidate)
        print(f'candidate panel:{[ch2stain[ch] for ch in candidate_panel]}')
        masked_ch_idx = torch.tensor([i for i in range(25) if i not in candidate_panel])

        with torch.no_grad():
            preds, masked = model.forward(val_data,  masked_patch_idx=masked_ch_idx)
        ssim_score, corr, var, _  = ssim_corr_eval(val_data, masked, preds)
        print(f'chosen  masking strategy: SSIM:{ssim_score} Spearman:{corr} variance:{var}')

        if corr > top_corr:
            print('found new top panel')
            top_corr = corr
            top_ch = ch_candidate

    top_panel.append(top_ch)
    print(f'optimal panel so far: {[ch2stain[ch] for ch in top_panel]}')
    with open(log,'a') as f:
        f.write(f'{[ch2stain[ch] for ch in top_panel]}\n')
        f.write(f'{top_panel}\n')
