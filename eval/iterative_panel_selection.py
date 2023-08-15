import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import torch
from torchmetrics.functional import spearman_corrcoef as spearman
from intensity import get_intensities
from data import get_data
from mae import IF_MAE


def get_channel_order(val_loader, model, ch2stain):
    device = model.device
    top_panel = [0]
    for num_masked in reversed(range(1,24)):
        masking_ratio = num_masked/25
        model.mae.masking_ratio = masking_ratio

        top_corr = -999
        top_ch = None
        for ch_candidate in range(25):
            if ch_candidate in top_panel: continue
            candidate_panel = top_panel.copy()
            candidate_panel.append(ch_candidate)
            print(f'candidate panel:{[ch2stain[ch] for ch in candidate_panel]}')

            mints, pmints = get_intensities(model, candidate_panel, val_loader, device=device)
            if mints.shape[1] == 1:
                mints = mints.squeeze()
                pmints = pmints.squeeze()
            corr = torch.mean(spearman(pmints, mints))

            if corr > top_corr:
                print('found new top panel')
                top_corr = corr
                top_ch = ch_candidate

            gc.collect()
            torch.cuda.empty_cache()

        top_panel.append(top_ch)
    return top_panel


if __name__ == '__main__':
    tissue = 'CRC' #BC
    device = torch.device('cuda:2')
    BATCH_SIZE = 2000

    if tissue == 'BC':
        ckpt = 'ckpts/BC_TMA_50p_mask.ckpt' #BC
        val_loader = get_data('BC', BATCH_SIZE)
        ch2stain = {0:"DAPI", 1:"CD3", 2:"ERK-1", 3:"hRAD51", 4:"CyclinD1", 5:"VIM", 6:"aSMA", 7:"ECad", 8:"ER", 9:"PR",
                    10:"EGFR", 11:"Rb", 12:"HER2", 13:"Ki67", 14:"CD45", 15:"p21", 16:"CK14", 17:"CK19", 18:"CK17",
                    19:"LaminABC", 20:"AR", 21:"Histone H2AX", 22:"PCNA", 23:"PanCK", 24:"CD31"}

    if tissue == 'CRC':
        ckpt = 'ckpts/CRC_TMA_50p_mask.ckpt'
        val_loader = get_data('CRC-TMA', BATCH_SIZE)
        ch2stain = {0:"DAPI", 1:"CD3", 2:"NaKATPase", 3:"CD45RO", 4:"Ki67", 5:"panCK", 6:"aSMA", 7:"CD4", 8:"CD45",
                    9:"PD-1", 10:"CD20", 11:"CD68", 12:"CD8a", 13:"CD163", 14:"FOXP3", 15:"PD-L1", 16:"ECad", 17:"Vim",
                    18:"CDX2", 19:"LaminABC", 20:"Desmin", 21:"CD31", 22:"PCNA", 23:"Ki67", 24:"Collagen IV"}

    model = IF_MAE()
    model = model.load_from_checkpoint(ckpt)
    model = model.to(device)
    model = model.eval()
    
    top_panel = get_channel_order(val_loader, model, ch2stain)
    print(f'found top panel: {top_panel}')
