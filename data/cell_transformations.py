import numpy as np
import cv2

#PREPROCESSING FUNCTIONS (taken from terneslu/PROJECTS/PanelReduction/SingleCellSegmentations.ipynb)
#rotate 
def rotate_image(image, angle):
    row,col,_ = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

#flip
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