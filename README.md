# A Masked Image Modelling Approach to Multiplex Tissue Imaging Panel Reduction

![IF-MAE](/src/mae_architecture.jpg "IF-MAE Architecture")


This is the repository that contains all code necessary to replicate the figures and experiments in the paper "PRIME: Panel Reduction and Imputation through Masked imaging modeling for Enhancing Cyclic Immunofluorescence (CyCIF)". 

# Dataset Access and construction
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
	10. Click the green button that says "Download link"
	11. in your downloads folder you should now have a file named "download-links.txt", to download, navigate to your Downloads directory in a terminal (`cd Downloads`), then enter `wget -i download-links.txt` to begin the download.
	12. the filenames will have a large amount of characters that come after the .ome.tif extension that needs to be removed, so you can do so by first creating a new directory and moving the files to that directory like so: `mkdir CRC-TMA`, then `mv HTMA4* CRC-TMA/` and finally, `find . -name '*.ome.tif*' -exec bash -c 'mv "$0" "${0%%ome.tif*}ome.tif"' {} \;`.
	
	
