from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import numpy as np
import torch
import os

def get_data(tissue_type, BATCH_SIZE):
    
    if tissue_type == 'BC':
        data_dir= '/var/local/ChangLab/panel_reduction/TMA4-CellTiles'
    elif tissue_type == 'CRC-TMA':
        #update this variable when constructing your own CRC-TMA dataset
        data_dir = '/var/local/ChangLab/panel_reduction/CRC-TMA-2'
    elif tissue_type == 'CRC-WSI':
        data_dir = '/var/local/ChangLab/panel_reduction/CRC-WSI'

    files = [f'{data_dir}/{f}' for f in os.listdir(data_dir)]

    split = round(len(files) * 0.9)
    val_files = files[split:]
    val_files = val_files[int(len(val_files)/2):]
    val_data = SingleCellDataset(val_files)
    if tissue_type == 'CRC-WSI':
        val_data = SingleCellDataset(files)
    
    #only load files with cells that are within ROIs adjacent to TMA punchouts
    #wsi_roi_cell_ids = list(np.load('../data/wsi_roi_cell_ids.npy'))
    #files = [f for f in files if int(f.split('-')[3].split('.')[0]) in wsi_roi_cell_ids]
    
    #only load files with cells in TMA cores that correspond to CRC02 WSI sample
    #matching_cores = ['HTMA402_2-099', 'HTMA402_2-098', 'HTMA402_2-097']
    #files = [f for f in files if any([c in f for c in matching_cores])]
    
    #val_data = SingleCellDataset(files)
        
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)
    return val_loader


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
        tensor = torch.from_numpy(im)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()

        return tensor, filepath.split('/')[-1]