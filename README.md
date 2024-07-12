In this project I tried to predict cancerous areas in biopsies using Deep Learning. This was my first attemp at something like this. 
The rename file renames the data files names to be in the same format.
The Data_Aug rotates and flips images to increase the amount of images.
The U-net_multi-class_DB1 is the main file that uses the rest of the files, after the rename and data_aug.

Here is the abstract of the report:

The goal of the project is to analyze histopathology images of the thyroid gland using deep learning algorithms. 
The algorithm is designed to reduce the workload of pathologists in identifying regions of interest in the biopsy samples and minimize human errors in the identification process.
I chose to perform the segmentation using the U-Net network.
Following the classification, I performed feature detection to classify the subtype of thyroid cancer using the StarDist network.
I conducted tests using datasets from two hospitals, NKI in the Netherlands and VGH in Canada, for the U-Net algorithm, and the MoNuSeg dataset for the StarDist network.
According to the IoU metric, I achieved an 80% correct identification rate on the U-Net network for the test set, and a 82% correct identification rate on the StarDist network.
