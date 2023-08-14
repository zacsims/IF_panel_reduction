import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow, imsave
from deepcell.applications import Mesmer
from mask_refinement import refine_masks
from channel_info import get_channel_info

keep_channels, keep_channels_idx, ch2idx = get_channel_info('CRC')
    
#load mesmer for segmentation
model = Mesmer()
data_dir = '/home/groups/ChangLab/dataset/CRC-TNP-TMA'
save_dir =  '/home/groups/ChangLab/simsz/panel_reduction/crc_tma_masks'

#load cores and their file names
cores = []
fnames = []
print('loading cores...')
for f in tqdm(os.listdir(data_dir)):
    core = imread(f'{data_dir}/{f}')
    if core.shape == (36, 1400, 1400): #there is 1 random core that has a different shape
        cores.append(core[keep_channels_idx])
        fnames.append(f)
        
print(f'segmenting cores')
#iterate through cores
for fname, core in tqdm(zip(fnames, cores)):
    sample_name = fname.split('.')[0]
    
    #get nuclei and cell morphology markers for input to segmentation model
    core_dapi = core[ch2idx['DAPI']]
    core_morph = np.max(core[[ch2idx['PanCK'], ch2idx['CD45']]], axis=0) 
    core_seg_in = np.expand_dims(np.stack([core_dapi, core_morph], axis=-1), axis=0)
    
    #segment cells with mesmer
    labeled_image = model.predict(core_seg_in, compartment='both', image_mpp=0.65, batch_size=1)
    cell_mask, nuc_mask = labeled_image[0,:,:,0], labeled_image[0,:,:,1]
    cell_mask, nuc_mask = refine_masks(cell_mask, nuc_mask)   
    
    save_fname = f'{sample_name}-mask.tif'
    imsave(os.path.join(save_dir, save_fname), cell_mask, check_contrast=False)                    
