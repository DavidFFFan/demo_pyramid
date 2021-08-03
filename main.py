import torchvision.transforms
from LPTN_arch import Lap_Pyramid_Conv, WORK_PATH
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
import argparse
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, dirname, transform=None):
        super(MyDataset, self).__init__()
        self.image_names = os.listdir(dirname)
        self.images = []
        self.transform = transform
        for name in self.image_names:
            self.images.append((os.path.join(dirname, name), name))  # 获得图片路径和类别名索引

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path, name = self.images[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='路径名称设置')
    parser.add_argument('path', help='输入的图片路径')
    parser.add_argument('--out_path', help='输出图片路径', default=os.path.join(WORK_PATH, 'output'))
    parser.add_argument('--names', help='不同分辨率的文件夹名称', type=str, nargs=4,
                        default=['level_1', 'level_2', 'level_3', 'level_4'])

    args = parser.parse_args()
    input_path = args.path
    out_path = args.out_path
    high_paths = args.names

    dataset = MyDataset(input_path, transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lap_pyramid = Lap_Pyramid_Conv(3).to(device)
    # log
    print('\ninput path:{}\nout path:{}\nlevel names:{}\n'.format(input_path, out_path, high_paths))
    print('start!')
    for image, name in tqdm(dataloader):
        image = image.to(device)
        pyr_A = lap_pyramid.pyramid_decom(img=image)
        for pos, img in enumerate(pyr_A):
            img = img.squeeze(0)
            img = img.permute(2, 1, 0)
            img = (img + 1.0) * 0.5
            img = img.cpu()
            img = img.mul(255).clamp(0, 255)
            img = np.array(img).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            path = os.path.join(WORK_PATH, out_path, high_paths[pos])

            if not os.path.exists(path):
                os.makedirs(path)

            pic_name = os.path.join(path, name[0])

            cv2.imwrite(pic_name, img)
    print('finished!\n')
