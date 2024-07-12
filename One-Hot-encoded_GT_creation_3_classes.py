import numpy as np
import os
import skimage.io as io
import skimage.transform as trans


def prepare_multi_class_GT(GT_PATH, class_names, savepath, target_size=(512, 512), n_class=3):
    f_names = os.listdir(GT_PATH + class_names[0])
    for files in f_names:
        GT_im = np.zeros(
            np.concatenate((target_size, n_class), axis=None))  # This creates a zero array of size (512,512,3)
        FG = np.zeros(target_size)

        for idx, cn in enumerate(class_names):
            lab = io.imread(GT_PATH + cn + files, as_gray=True)
            lab = trans.resize(lab, target_size)
            if (np.max(lab) > 1):
                lab = lab / 255
            lab[lab >= 0.1] = 1  # threshold at 0.1. Change this value based on your requirement
            lab[lab < 0.1] = 0

            if (idx < 2):  # Bright Lesions
                GT_im[:, :, 1] = GT_im[:, :, 1] + lab
            else:  # Red Lesions
                GT_im[:, :, 0] = GT_im[:, :, 0] + lab
        if (np.sum(GT_im[:, :, 0]) > 0):
            GT_im[:, :, 0] = GT_im[:, :, 0] / np.ptp(GT_im[:, :, 0])
        if (np.sum(GT_im[:, :, 1]) > 0):
            GT_im[:, :, 1] = GT_im[:, :, 1] / np.ptp(GT_im[:, :, 1])
        FG = (GT_im[:, :, 0] + GT_im[:, :, 1] > 0).astype(int)
        GT_im[:, :, 2] = 1 - FG
        io.imsave(savepath + files, GT_im)

        GT_PATH = './ddb1_groundtruth/'
        class_names = ['hemorrhages/', 'redsmalldots/', 'hardexudates/', 'softexudates/', ]

        prepare_multi_class_GT(GT_PATH, class_names, './GT/')