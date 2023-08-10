import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow, imsave
from deepcell.applications import Mesmer
from mask_refinement import refine_masks



#####################################################################################

'''
#CRC TMA channel order
channels = ["DAPI", "Control", "Control", "Control",
            "DAPI_2", "CD3", "NaKATPase", "CD45RO",
            "DAPI_3", "Ki67_1", "PanCK", "aSMA",
            "DAPI_4", "CD4", "CD45", "PD-1",
            "DAPI_5", "CD20", "CD68", "CD8a",
            "DAPI_6", "CD163", "FOXP3", "PD-L1",
            "DAPI_7", "ECad", "Vimentin", "CDX2",
            "DAPI_8", "LaminABC", "Desmin", "CD31",
            "DAPI_9", "PCNA", "Ki67_2", "Collagen IV"]

#order of 17 channel panel to match ORION dataset plus last round dapi to check for retention
keep_channels = ["DAPI", "CD3", "CD45RO", "Ki67_1", "PanCK", "CD4", "CD45", "PD-1", 
                 "CD20", "CD68", "CD8a", "CD163", "FOXP3", "PD-L1", "ECad","CD31",
                 "DAPI_9"]
'''
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

keep_channels = ["DAPI"] + [ch for ch in channels if ch != "Control" and not ch.startswith('DAPI')] + ["DAPI_9"]
keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
ch2idx = {ch:i for i,ch in enumerate(keep_channels)}
    
#load mesmer for segmentation
model = Mesmer()
data_dir = '/home/groups/ChangLab/dataset/CRC-TNP-TMA/'
save_dir =  '/home/groups/ChangLab/simsz/panel_reduction/crc_tma_masks'

#load cores and their file names
cores = []
fnames = []
print('loading cores...')
for f in tqdm(os.listdir('/home/groups/ChangLab/dataset/CRC-TNP-TMA')):
    core = imread(f'/home/groups/ChangLab/dataset/CRC-TNP-TMA/{f}')
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