import glob
import random
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

class MultiResolutionDataset(Dataset):
    def __init__(self, dataset_folder, dataset_type, transform, nerf_resolution=64):
        import time
        t0 = time.time()
        print('Start loading file addresses ...')
        images = glob.glob(dataset_folder + '/*.' + dataset_type)
        random.shuffle(images)
        t = time.time() - t0
        print('done! time:', t)
        print("Number of images found: %d" % len(images))
        self.images = images
        self.length = len(images)
        self.nerf_resolution = nerf_resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        buffer = self.images[index]
        img = Image.open(buffer)
        if random.random() > 0.5:
            img = TF.hflip(img)

        thumb_img = img.resize((self.nerf_resolution, self.nerf_resolution), Image.HAMMING)
        img = self.transform(img)
        thumb_img = self.transform(thumb_img)

        return img, thumb_img