import os
from tqdm import tqdm
import numpy as np
from skimage.io import imread, imsave
from skimage.exposure import match_histograms
from einops import rearrange
from channel_info import get_channel_info


keep_channels, keep_channels_idx, ch2idx = get_channel_info('CRC')

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