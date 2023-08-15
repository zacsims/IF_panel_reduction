from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
import numpy as np
from mask_refinement import refine_masks
from skimage.measure import regionprops
from einops import repeat,rearrange
import math
from tqdm import tqdm
import tifffile
import os
from channel_info import get_channel_info
from cell_transformations import flip_mask, rotate_image


#normalization
def normalize(wsi):
    increase_min = [1, 7, 8, 9, 10, 13, 14, 15, 16, 17, 21, 23]
    output = np.empty(wsi.shape, dtype='uint8')
    for i,_ in tqdm(enumerate(wsi)):
        
        ch = wsi[i].copy()
        min_percentile = 1
        min_ = np.percentile(ch, min_percentile)
        ch =ch.astype('float') - min_
        ch[ch < 0]=0

        ch=(ch * 255 / np.percentile(ch, 99))
        ch[ch >= 255] = 255
        ch = ch.astype('uint8')
        output[i] = ch
        
    return output

keep_channels, keep_channels_idx, ch2idx = get_channel_info('CRC')

data_dir = '/home/groups/ChangLab/dataset/HMS-CRC-WSI'
save_dir =  '/var/local/ChangLab/panel_reduction/CRC-WSI'
mask_save_dir = '/var/local/ChangLab/panel_reduction/CRC-WSI-cell-masks'

sample_name = 'CRC02'
wsi = imread('crc02_matched.tif')
print('normalizing wsi...')
wsi = normalize(wsi)

print(f'extracting cells from wsi')
cell_mask = imread(os.path.join(data_dir, 'CRC02_mesmer_cell_mask.tif'))

pad = ((0,),(16,), (16,)) #pad 2nd and 3rd dimensions with 16 pixels
wsi, cell_mask = np.pad(wsi, pad), np.pad(cell_mask, 16)

#iterate through cell regions
rps = regionprops(cell_mask.astype('int'))
for rp in tqdm(rps):

    #size filter
    if rp.area > 400 or rp.area < 30: continue

    #get centroid of cell and and bbox by expanding out in each direction by 16 pixels to obtain 32x32px image
    center_x, center_y = int(rp.centroid[0]), int(rp.centroid[1])
    xmin, xmax, ymin, ymax = center_x - 16, center_x + 16, center_y - 16, center_y + 16

    #crop image and mask
    im = wsi[:, xmin:xmax, ymin:ymax].copy()
    mask = cell_mask[xmin:xmax, ymin:ymax].copy()

    #isolate single cell
    mask[mask != rp.label] = 0 
    #repeat mask for all channels
    mask = repeat(mask, 'h w -> c h w', c=len(keep_channels))
    #zero out background
    im[mask == 0] = 0 

    #check for segmentation error
    if im[0].mean() == 0: continue 
        
    #check for last round dapi retention
    last_dapi = im[ch2idx['DAPI_9']]
    if np.mean(last_dapi) == 0:
        #print('dapi retention')
        continue

    #move channel dimension
    im = np.moveaxis(im, 0, 2)
    mask = np.moveaxis(mask, 0, 2)

    #perform transformations
    im = rotate_image(im,-math.degrees(rp.orientation))
    mask = rotate_image(mask, -math.degrees(rp.orientation))
    im, mask = flip_mask(im, mask)

    #save image
    im = im[:,:,:-1] #dont need last round dapi
    save_fname = f'{sample_name}-CellID-{rp.label}.tif'
    save_mask_fname = f'{sample_name}-CellID-{rp.label}-mask.tif'
    imsave(os.path.join(save_dir, save_fname), im.astype('uint8') , check_contrast=False)        
    imsave(os.path.join(mask_save_dir, save_mask_fname), mask, check_contrast=False) 