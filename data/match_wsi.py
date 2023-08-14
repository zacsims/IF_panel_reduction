import os
from tqdm import tqdm
import numpy as np
from skimage.io import imread, imsave
from skimage.exposure import match_histograms
from einops import rearrange



channels = [
    #Round 0
    "DAPI",
    "Control",
    "Control",
    "Control",
    #Round 1
    "DAPI_2",
    "CD3",
    "NaKATPase",
    "CD45RO",
    #Round 2
    "DAPI_3",
    "Ki67",
    "PanCK",
    "aSMA",
    #Round 3
    "DAPI_4",
    "CD4",
    "CD45",
    "PD-1/CD279",
    #Round 4
    "DAPI_5",
    "CD20",
    "CD68",
    "CD8a",
    #Round 5
    "DAPI_6",
    "CD163",
    "FOXP3",
    "PD-L1/CD274",
    #Round 6
    "DAPI_7",
    "E-Cadherin",
    "Vimentin",
    "CDX2",
    #Round 7
    "DAPI_8",
    "LaminABC",
    "Desmin",
    "CD31",
    #Round 8
    "DAPI_9",
    "PCNA",
    "Ki67",
    "Collagen IV",
]

#collect channel info
keep_channels = ["DAPI"] + [ch for ch in channels if ch != "Control" and not ch.startswith('DAPI')] + ["DAPI_9"]
keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
ch2idx = {ch:i for i,ch in enumerate(keep_channels)}

#load wsi
wsi = imread('/home/groups/ChangLab/dataset/HMS-CRC-WSI/CRC02.ome.tif')[keep_channels_idx]
#get names of cores in TMA that were collected from same tissue section as WSI
matching_cores = ['HTMA402_2-099.ome.tif', 'HTMA402_2-098.ome.tif', 'HTMA402_2-097.ome.tif']
data_dir = '/home/groups/ChangLab/dataset/CRC-TNP-TMA/'
#load cores and their file names
cores = []
fnames = []
print('loading cores...')
for f in tqdm(os.listdir('/home/groups/ChangLab/dataset/CRC-TNP-TMA')):
    core = imread(f'/home/groups/ChangLab/dataset/CRC-TNP-TMA/{f}')
    if core.shape == (36, 1400, 1400): #there is 1 random core that has a different shape
        if f.split('/')[-1] in matching_cores:
            cores.append(core[keep_channels_idx])
            fnames.append(f)


cores_flat = np.concatenate([rearrange(c, 'c h w -> c (h w)') for c in cores], axis=1)

def match_wsi(wsi, cores):
    """matches WSI channel histograms to TMA cores"""
    output = np.empty(wsi.shape, dtype='uint16')
    for i,ch in tqdm(enumerate(wsi)):
        output[i] = match_histograms(ch.flatten(), cores[i]).reshape(wsi[0].shape)
    return output

wsi_matched = match_wsi(wsi, cores_flat)
imsave('crc02_matched.tif', wsi_matched)