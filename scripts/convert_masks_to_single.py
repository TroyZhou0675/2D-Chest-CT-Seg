from PIL import Image
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def Convert_Multiple_masks_to_Single_mask(mask,channel_of_reserved_mask=3):
    return mask==channel_of_reserved_mask



def convert_folder(src_dir, dst_dir,channel):
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if not name.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        # 打开并转RGB
        img = Image.open(os.path.join(src_dir, name))


        img_np = np.array(img)
        # 转为索引图
        img_singlemask_np = Convert_Multiple_masks_to_Single_mask(img_np,channel_of_reserved_mask=channel)
        # 保存为灰度
        Image.fromarray(img_singlemask_np, mode='L').save(os.path.join(dst_dir, os.path.splitext(name)[0] + '.png'))
    print("转换完成！")

src_dir="D:/0-MyDoc/DeepLearning/MIS/CT_SEG/data/processed/masks/"
dst_dir="D:/0-MyDoc/DeepLearning/MIS/CT_SEG/data/processed/mask_green/"

convert_folder(src_dir=src_dir,dst_dir=dst_dir,channel=2)