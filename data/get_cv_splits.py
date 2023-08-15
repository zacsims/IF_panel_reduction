import os
import pickle
from skimage.io import imread
from tqdm import tqdm

def get_cv_splits():
    '''splits CRC TMA core dataset into 5 cross-validation sets and saves core filenames in picke files'''
    fnames = []
    print('loading cores...')
    outlier_set = [40, 72, 81, 86, 98, 114, 130, 135, 146, 267, 288, 330]
    for i,f in tqdm(enumerate(os.listdir('/home/groups/ChangLab/dataset/CRC-TNP-TMA'))):
        core = imread(f'/home/groups/ChangLab/dataset/CRC-TNP-TMA/{f}')
        if core.shape == (36, 1400, 1400): #there is 1 random core that has a different shape
            if i not in outlier_set:
                fnames.append(f)
    fnames = [f.split('.')[0] for f in fnames]           
    splits = [(0,64), (64,128), (128, 192), (192,256), (256,319)]
    s = 1
    for i,j in splits:
        test_set = fnames[i:j]
        train_set = [f for f in fnames if f not in test_set]
        with open(f'cv_{s}_cores.pkl','wb') as f:
            pickle.dump(train_set, f)
        s += 1