import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate

"""Create a directory"""
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "train", "image", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "mask", "*.jpg")))

    test_x = sorted(glob(os.path.join(path, "test", "image", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "test", "mask", "*.jpg")))

    return (train_x,train_y), (test_x,test_y)

def augment_data(images, masks ,save_path, augment=True):
    size= (512,512)

    for idx, (x, y) in  tqdm(enumerate(zip(images, masks)), total=len(images)):
        """Extracting the name"""
        name = x.split("/")[-1].split(".")[-2]
        print(name)

        """Reading image and mask"""
        x = cv2.imread(x,cv2.IMREAD_COLOR);
        #cv2.imshow("hi",x);
        #cv2.waitKey()
        #break
        #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        y = imageio.mimread(y)[0]

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=90, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
                i = cv2.resize(i, size)
                m = cv2.resize(m, size)

                tmp_image_name = f"{name}_{index}.png"
                tmp_mask_name = f"{name}_{index}.png"

                image_path = os.path.join(save_path, "image", tmp_image_name)
                mask_path = os.path.join(save_path, "mask", tmp_mask_name)

                cv2.imwrite(image_path, cv2.cvtColor(i,cv2.COLOR_RGB2BGR))
                cv2.imwrite(mask_path, cv2.cvtColor(m,cv2.COLOR_RGB2BGR))

                index += 1

if __name__ == "__main__":
    """Seeding"""
    np.random.seed(42)

    """Load the Data"""
    data_path = "/home/lior.go@staff.technion.ac.il/PycharmProjects/pythonProject/nki_vgh_data"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """Create directries to save the augmented data"""
    create_dir("new_data_biopsy/train/image/")
    create_dir("new_data_biopsy/train/mask/")
    create_dir("new_data_biopsy/test/image/")
    create_dir("new_data_biopsy/test/mask/")

    """Data augmentation"""
    augment_data(train_x, train_y, "new_data_biopsy/train/", augment=True)
    augment_data(test_x, test_y, "new_data_biopsy/test/", augment=False)