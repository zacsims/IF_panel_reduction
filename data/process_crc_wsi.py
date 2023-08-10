from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
import numpy as np
from mask_refinement import refine_masks
from skimage.measure import regionprops
from einops import repeat,rearrange
import cv2
import math
from tqdm import tqdm
import tifffile
import os


#PREPROCESSING FUNCTIONS (taken from terneslu/PROJECTS/PanelReduction/SingleCellSegmentations.ipynb)
#rotate 
def rotateImage(image, angle):
    row,col,_ = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def flip_mask(image, mask):
    #Identify quadrant
    up_left = np.mean(image[0,:16,:16][image[0,:16,:16]>0])
    down_left = np.mean(image[0,16:,:16][image[0,16:,:16]>0])
    up_right = np.mean(image[0,:16,16:][image[0,:16,16:]>0])
    down_right = np.mean(image[0,16:,16:][image[0,16:,16:]>0])
    
    vec = [up_left, down_left, up_right, down_right]
    index = np.argmax(vec)
    
    if index == 0:
        FlippedImage = image
        FlippedMask = mask
    elif index == 1:
        FlippedImage = np.flipud(image)
        FlippedMask = np.flipud(mask)
    elif index == 2:
        FlippedImage = np.fliplr(image)
        FlippedMask = np.fliplr(mask)
    elif index == 3:
        FlippedImage = np.fliplr(image)
        FlippedImage = np.flipud(FlippedImage)
        FlippedMask = np.fliplr(mask)
        FlippedMask = np.flipud(FlippedMask)
    
    return FlippedImage, FlippedMask


#normalization
def normalize(wsi):
    increase_min = [1, 7, 8, 9, 10, 13, 14, 15, 16, 17, 21, 23]
    output = np.empty(wsi.shape, dtype='uint8')
    for i,_ in tqdm(enumerate(wsi)):
        
        ch = wsi[i].copy()
        #min_ = ch.min()
        min_percentile = 1
        #if i in increase_min: min_percentile = 10
        min_ = np.percentile(ch, min_percentile)
        ch =ch.astype('float') - min_
        ch[ch < 0]=0

        ch=(ch * 255 / np.percentile(ch, 99))
        ch[ch >= 255] = 255
        ch = ch.astype('uint8')
        output[i] = ch
        
    return output

#####################################################################################


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

data_dir = '/home/groups/ChangLab/dataset/HMS-CRC-WSI'
save_dir =  '/var/local/ChangLab/panel_reduction/CRC-WSI'
mask_save_dir = '/var/local/ChangLab/panel_reduction/CRC-WSI-cell-masks'

sample_name = 'CRC02'
fname = 'CRC02.ome.tif'
#wsi = imread(os.path.join(data_dir, fname))
#wsi = imread('/home/groups/ChangLab/simsz/panel_reduction/data/crc02_matched.tif')
#wsi = wsi[keep_channels_idx]        
#normalize
#print('normalizing wsi...')
#wsi = normalize(wsi)
#imsave('crc02_matched_256b.tif', wsi)
wsi = imread('crc02_matched_256b.tif')

print(f'extracting cells from wsi')
sample_name = fname.split('.')[0]

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
    im = rotateImage(im,-math.degrees(rp.orientation))
    mask = rotateImage(mask, -math.degrees(rp.orientation))
    im, mask = flip_mask(im, mask)

    #save image
    im = im[:,:,:-1] #dont need last round dapi
    save_fname = f'{sample_name}-CellID-{rp.label}.tif'
    save_mask_fname = f'{sample_name}-CellID-{rp.label}-mask.tif'
    imsave(os.path.join(save_dir, save_fname), im.astype('uint8') , check_contrast=False)        
    imsave(os.path.join(mask_save_dir, save_mask_fname), mask, check_contrast=False) 