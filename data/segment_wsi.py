from skimage.io import imread, imsave
from deepcell.applications import Mesmer
import numpy as np
import gc
from mask_refinement import refine_masks

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


keep_channels = ["DAPI"] + [ch for ch in channels if ch != "Control" and not ch.startswith('DAPI')]
keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
ch2idx = {ch:i for i,ch in enumerate(keep_channels)}

wsi = imread('/home/groups/ChangLab/dataset/HMS-CRC-WSI/CRC02.ome.tif')

split = wsi.shape[2] // 2
wsi_1 = wsi[:,:,:split]
wsi_2 = wsi[:,:,split:]

del wsi
gc.collect()

wsis = [wsi_1, wsi_2]
cell_masks = []
nuc_masks = []

for wsi in wsis:
    dapi_ch = wsi[ch2idx['DAPI']] #get DAPI Channel for Nucleus
    morph_ch = np.max(wsi[[ch2idx['PanCK'], ch2idx['CD45']]], axis=0) # Get PanCK + CD45 channels for Membrane
    input_ = np.expand_dims(np.stack([dapi_ch, morph_ch], axis=-1), axis=0) #stack input channels

    #segment cells with mesmer
    model = Mesmer()
    labeled_image = model.predict(input_, compartment='both', image_mpp=0.65, batch_size=1)
    cell_mask, nuc_mask = labeled_image[0,:,:,0], labeled_image[0,:,:,1]
    cell_mask_, nuc_mask_ = refine_masks(cell_mask, nuc_mask)
    
    cell_masks.append(cell_mask_)
    nuc_masks.append(nuc_mask_)
    
max_cell_id = np.max(cell_masks[0])
cell_masks[1] += max_cell_id

max_nuc_id = np.max(nuc_masks[0])
nuc_masks[1] += max_nuc_id
    
cell_mask = np.concatenate(cell_masks, axis=1)
nuc_mask = np.concatenate(nuc_masks, axis=1)

imsave('/home/groups/ChangLab/dataset/HMS-CRC-WSI/CRC02_mesmer_cell_mask.tif', cell_mask)
imsave('/home/groups/ChangLab/dataset/HMS-CRC-WSI/CRC02_mesmer_nuc_mask.tif', nuc_mask)
