# A Masked Image Modelling Approach to Multiplex Tissue Imaging Panel Reduction

![IF-MAE](/src/mae_arch.png "IF-MAE Architecture")


This is the repository that contains all code necessary to replicate the figures and experiments in the paper "MIM-CyCIF: Masked Imaging Modeling for Enhancing Cyclic Immunofluorescence (CyCIF) with Panel Reduction and Imputation". The goal of this project is to train a model capable of generating multiplex proteomic data in pixel space from a small subset of markers, enabling larger panels to be used, or for high-plex data to be generated from data collected using more traditional immunofluorescence methods. For example, we show that a 25 marker CyCIF panel can be reliably reconstructed from just 9 markers. 

# Dataset Access and Construction
## Data Download
To access the CRC-TMA dataset, the easiest way is to go through the [Cancer Genomics Cloud](https://cgc-accounts.sbgenomics.com). To do so, follow these steps:
1. Login or create an account
2. Once logged in, find the green button that says 'Create Project' and create a new project with the default settings.
3. At the top of the page, under the "Data" tab, select "Cancer Data Service Explorer"
4. Click "Explore Files"
5. In the "Search files by File name" box, enter "HTMA4"
6. In the top right of the screen, select "Copy to project"
7. select the project you created in step 2. and click "Copy"
8. In the top left above the filenames, click the black arrow pointing down and click "select all"
9. at the top, next to the "Download" button, select the black arrow pointing down and click "Get download links"
10. Click the green button that says "Download link"	11. in your downloads folder you should now have a file named "download-links.txt", to download, navigate to your Downloads directory in a terminal (`cd Downloads`), then enter `wget -i download-links.txt` to begin the download.
12. the f;ilenames will have a large amount of characters that come after the .ome.tif extension that needs to be removed, so you can do so by first creating a new directory and moving the files to that directory like so: `mkdir CRC-TMA`, then `mv HTMA4* CRC-TMA/` and finally, `find . -name '*.ome.tif*' -exec bash -c 'mv "$0" "${0%%ome.tif*}ome.tif"' {} \;`.
## Create Single-Cell Data
One you have downloaded the cores, you can begin running the scripts to process the data into individual singe-cells. 
1. the first script is the segmentation script: `data/segment_crc_tma.py`. Before running, you will need to update the `data_dir` directory to the directory you saved the TMA cores in, and update the `save_dir` directory to wherever you would like the segmentation masks for each core to be saved.
2. next you can run the processing script `data/process_crc_tma.py`. again you will first have to update the `data_dir` variable to point to the TMA cores, the `mask_dir` variable to be where you saved the segmentation masks, `save_dir` to be where you would like the single-cell images to be stored, and `mask_save_dir` to be where the single-cell masks are saved.	

## Model 
### Training
The model training script for the implementation described in the paper is `training/run_mae.py`, and the model implementation is in `training/mae.py`. Before running you will need to update the `data_dir` variable to point to the directory containing the single-cell images. Note that you do not need to train a model yourself to replicate our experiments since we provide model checkpoints in `eval/ckpts`. A seperate training script for cross-validation is provided.	
### Inference
Inside the `eval/ckpts` directory we provide the checkpoints for 4 trained models, 3 models trained on the Breast Cancer TMA, each with a different masking ratio (25%, 50%, 75%), and one model trained on the Colorectal Cancer TMA at a 50% masking ratio. We find that models trained at a 50% masking ratio perform the best.

## Panel Selection

The iterative marker selection procedure is done in `eval/iterative_panel_selection.py`. Running this script will print a list containing an ordering of channel indices that can then be input into other evaluation notebooks later on. 

![IPS](/src/iterative_selection_ex.gif "Iterative Panel Selection example")

## Model Evaluation
All model evaluation is done inside of the notebooks in the `eval` directory. Prior to running these notebooks, you will need to update the `data_dir` variable in `eval/data.py` to point to the correct directory on your machine containing the single cell images and also update the `dir_` variable in `eval/intensity.py` to point to the directory of the single cell segmentation masks . All models were evaluated using a single Nvidia A40 GPU but should be runnable on any machine by adjusting the `device` and `BATCH_SIZE` variables at the top of each notebook. The corresponding notebook that generates each figure in the paper is as follows:
- Figure 1B: `MAE Figures - RGB reconstructions.ipynb`
- Figures 2A,2C, and Supplementary Figure 1: `mean intensity spearman correlation-BC.ipynb` 
- Figures 2B and Supplementary Figure 6: `MAE Figures - comparison to ME-VAE.ipynb`
- Supplementary Figures 1,2, and 3: `mean intensity spearman correlation-CRC.ipynb`
- Supplementary Figure 4: `MAE Figures - CRC Cross Validation.ipynb`

## Panel Information

Below is a table outlining which markers are included in the panels used to train the BC and CRC models. The 25 marker panels consist of the first round DAPI plus the 3 other markers in every cycle.

### Breast Cancer CyCIF Panel

|             |Channel 1 | Channel 2 | Channel 3 | Channel 4|
| ----------- | ---------|-----------|-----------|---------|
| Cycle 1     | DAPI | CD3 | ERK-1 | hRAD51 |
| Cycle 2     | DAPI | CyclinD1 | Vim | aSMA |
| Cycle 3     | DAPI | ECad | ER | PR |
| Cycle 4     | DAPI | EGFR | Rb | HER2 |
| Cycle 5     | DAPI | Ki67 | CD45 | p21 |
| Cycle 6     | DAPI | CK14 | CK19 | CK17|
| Cycle 7     | DAPI |LaminABC| AR| Histone H2AX |
| Cycle 8     | DAPI | PCNA | PanCK | CD31 |

### Colorectal Cancer CyCIF Panel

|             |Channel 1 | Channel 2 | Channel 3 | Channel 4|
| ----------- | ---------|-----------|-----------|---------|
| Cycle 1     | DAPI | CD3 | NaKATPase | CD45RO |
| Cycle 2     | DAPI | Ki67 | PanCK | aSMA |
| Cycle 3     | DAPI | CD4 | CD45 | PD-1|
| Cycle 4     | DAPI | CD20 | CD68 | CD8a |
| Cycle 5     | DAPI | CD163 | FOXP3 | PD-L1 |
| Cycle 6     | DAPI | ECad | Vim | CDX2|
| Cycle 7     | DAPI |LaminABC| Desmin| CD31 |
| Cycle 8     | DAPI | PCNA | Ki67 | Collagen IV |