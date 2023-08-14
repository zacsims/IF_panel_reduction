from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label
from einops import repeat,rearrange
import math
from tqdm import tqdm
import os
from channel_info import get_channel_info
from cell_transformations import flip_mask, rotate_image

#normalization
def normalize(cores, keep_channels):
    #flatten and concatenate cores
    cores = np.concatenate([rearrange(c, 'c h w -> c (h w)') for c in cores], axis=1)

    output = np.empty(cores.shape, dtype='uint8')
    for i,_ in tqdm(enumerate(cores)):
        ch = cores[i].copy()
        min_ = np.percentile(ch[ch > 0], 1)
        #max_ = np.percentile(ch[ch > 0], 99)
        ch =ch.astype('float') - min_
        ch[ch < 0]=0
        max_ = np.percentile(ch[ch > 0], 99)
        ch=(ch * 255 / max_)
        ch[ch >= 255] = 255
        ch = ch.astype('uint8')
        output[i] = ch
        print(f'{keep_channels[i]}: {min_=} {max_=}')
        
    return rearrange(output, 'c (n h w) -> n c h w', h=1400, w=1400)


keep_channels, keep_channels_idx, ch2idx = get_channel_info('CRC')
    
data_dir = '/home/groups/ChangLab/dataset/CRC-TNP-TMA'
mask_dir = '/home/groups/ChangLab/simsz/panel_reduction/data/crc_tma_masks'
save_dir = '/var/local/ChangLab/panel_reduction/CRC-TMA-2'
mask_save_dir = '/var/local/ChangLab/panel_reduction/CRC-TMA-2-cell-masks'

#load cores and their file names
cores_ = []
fnames_ = []
print('loading cores...')
for f in tqdm(os.listdir(data_dir)):
    core = imread(f'{data_dir}/{f}')
    if core.shape == (36, 1400, 1400): #there is 1 random core that has a different shape
        cores_.append(core[keep_channels_idx])
        fnames_.append(f)
cores = []
fnames = []
outlier_set = [40, 72, 81, 86, 98, 114, 130, 135, 146, 267, 288, 330]
for i, (core,fname) in enumerate(zip(cores_, fnames_)):
    if i not in outlier_set:
        cores.append(core)
        fnames.append(fname)
        
print(f'loaded {len(cores)} cores')       
#normalize
print('normalizing cores...')
cores = normalize(cores, keep_channels)

print(f'extracting cells from cores')
#iterate through cores
shape_statistics = {}
for fname, core in tqdm(zip(fnames, cores)):
    sample_name = fname.split('.')[0]
    
    cell_mask = imread(f'{mask_dir}/{sample_name}-mask.tif')
    pad = ((0,), (16,), (16,)) #pad 2nd and 3rd channels with 16 pixels
    core, cell_mask = np.pad(core, pad), np.pad(cell_mask, 16)

    #iterate through cell regions
    rps = regionprops(cell_mask.astype('int'))
    print(f'found {len(rps)} cells')
    for rp in rps:
        #size filter
        if rp.area > 400 or rp.area < 30:
            #print('size filter')
            continue
        
        #get centroid of cell and and bbox by expanding out in each direction by 16 pixels to obtain 32x32px image
        center_x, center_y = int(rp.centroid[0]), int(rp.centroid[1])
        xmin, xmax, ymin, ymax = center_x - 16, center_x + 16, center_y - 16, center_y + 16

        #crop image and mask
        im = core[:, xmin:xmax, ymin:ymax].copy()
        mask = cell_mask[xmin:xmax, ymin:ymax].copy()
        
        #isolate single cell
        mask[mask != rp.label] = 0 
        mask[mask > 0] = 1 
        #repeat mask for all channels
        mask = repeat(mask, 'h w -> c h w', c=len(keep_channels))
        #zero out background
        im[mask == 0] = 0 
       
        #check for segmentation error
        if im[0].mean() == 0:
            #print('seg error')
            continue 

        #check for last round dapi retention
        last_dapi = im[ch2idx['DAPI_9']]
        if np.mean(last_dapi) == 0:
            #print('dapi retention')
            continue
            
        #shape_stats = {'area':rp.area, 'area_bbox':rp.area_bbox, 'axis_major_length':rp.axis_major_length}
        #shape_statistics[f'{sample_name}-CellID-{rp.label}'] = shape_stats 
        
        #move channel dimension
        im = np.moveaxis(im, 0, 2)
        mask = np.moveaxis(mask, 0, 2)
        #perform transformations
        im = rotate_image(im, -math.degrees(rp.orientation))
        mask = rotate_image(mask, -math.degrees(rp.orientation))
        im, mask = flip_mask(im, mask)

        assert mask.sum() != 0
        #save image
        im = im[:,:,:-1] #dont need last round dapi
        save_fname = f'{sample_name}-CellID-{rp.label}.tif'
        save_mask_fname = f'{sample_name}-CellID-{rp.label}-mask.tif'
        imsave(os.path.join(save_dir, save_fname), im.astype('uint8') , check_contrast=False) 
        imsave(os.path.join(mask_save_dir, save_mask_fname), mask.astype('uint8'), check_contrast=False) 
        
#import pickle 
#with open('crc_tma_shape_statistics.pkl', 'wb') as f:
    #pickle.dump(shape_statistics, f)
