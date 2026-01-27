import sys
sys.path.append(r'D:/0-MyDoc/DeepLearning/MIS/CT_SEG/')
from src import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import skimage.io as io
import skimage.transform as trans
import os

IMG_SIZE = (256, 256, 1)
BATCH_SIZE = 8
NUM_CLASSES = 4
model_path = 'D:/0-MyDoc/DeepLearning/MIS/CT_SEG/experiments/2025-9-11-1/best_unet.h5'
model =unet()
model.load_weights(model_path)

image_partition, mask_partition = get_partitions(
    images_dir='D:/0-MyDoc/DeepLearning/MIS/CT_SEG/data/processed/images/',
    masks_dir='D:/0-MyDoc/DeepLearning/MIS/CT_SEG/data/processed/masks/'
)

val_gen = CustomDataGenerator(
    image_filenames=image_partition['test'],
    mask_filenames=mask_partition['test'],
    batch_size=BATCH_SIZE,
    dim=IMG_SIZE,
    n_classes=NUM_CLASSES,
    augment=False,
    shuffle=False
)


def calculate_segmentation_metrics(y_true, y_pred, num_classes=4, ignore_background=True):
    """
    计算多分类的 Dice 和 IoU 指标
    :param y_true: 真实标签，形状为 (N, H, W) 或 (N, H, W, 1)
    :param y_pred: 模型预测结果（Softmax后的概率），形状为 (N, H, W, num_classes)
    :param num_classes: 类别数（包含背景）
    :param ignore_background: 计算平均值时是否忽略背景类
    :return: 包含各项指标的字典
    """
    
    # 1. 将预测概率图转换为类别索引图 (N, H, W)
    y_pred_label = np.argmax(y_pred, axis=-1)
    
    # 2. 确保 y_true 也是 (N, H, W) 形状的索引图
    if y_true.ndim == 4:
        y_true = np.squeeze(y_true, axis=-1)
    
    dice_list = []
    iou_list = []
    
    print(f"{'Class':<10} | {'Dice':<10} | {'IoU':<10}")
    print("-" * 35)

    # 3. 逐类计算
    for c in range(num_classes):
        # 创建当前类的二值掩码
        true_c = (y_true == c)
        pred_c = (y_pred_label == c)
        
        intersection = np.logical_and(true_c, pred_c).sum()
        union = np.logical_or(true_c, pred_c).sum()
        target_sum = true_c.sum() + pred_c.sum()
        
        # 计算 IoU (Jaccard Index)
        if union == 0:
            iou = 1.0  # 如果真实和预测都没有该类，视为完美匹配
        else:
            iou = intersection / union
            
        # 计算 Dice 系数 (F1-score)
        if target_sum == 0:
            dice = 1.0
        else:
            dice = (2. * intersection) / target_sum
            
        dice_list.append(dice)
        iou_list.append(iou)
        
        print(f"Class {c:<4} | {dice:<10.4f} | {iou:<10.4f}")

    # 4. 计算平均指标
    start_idx = 1 if ignore_background else 0
    mDice = np.mean(dice_list[start_idx:])
    mIoU = np.mean(iou_list[start_idx:])
    
    print("-" * 35)
    print(f"Mean (excl. BG if set): Dice = {mDice:.4f}, IoU = {mIoU:.4f}")
    
    return {
        "dice_per_class": dice_list,
        "iou_per_class": iou_list,
        "mDice": mDice,
        "mIoU": mIoU
    }

if __name__ == "__main__":
    # metrics = calculate_segmentation_metrics(y_true, results, num_classes=4)
    # ==========================================
    # 1. 运行预测
    # ==========================================
    print("正在进行预测...")
    results = model.predict(val_gen, verbose=1)
    #results = model.predict(val_gen)  # 形状为 (N, 256, 256, NUM_CLASSES)

    # ==========================================
    # 2. 提取验证集的真实标签 (Ground Truth)
    # ==========================================
    print("正在提取真实标签...")
    all_y_true = []

    # 遍历生成器，获取所有的 y (label)
    for i in range(len(val_gen)):
        _, y_batch = val_gen[i]  # val_gen[i] 返回 (X_batch, y_batch)
        all_y_true.append(y_batch)

    # 将所有 batch 合并为一个大的 numpy 数组
    y_true_onehot = np.concatenate(all_y_true, axis=0)

    # 如果你的 calculate_segmentation_metrics 函数需要索引格式 (N, H, W)
    # 而 generator 输出的是 One-hot 格式 (N, H, W, C)，则需要转换：
    y_true_idx = np.argmax(y_true_onehot, axis=-1)

    # ==========================================
    # 3. 计算并展示指标
    # ==========================================
    # 注意：这里 num_classes 使用你定义的 NUM_CLASSES
    metrics = calculate_segmentation_metrics(
        y_true=y_true_idx, 
        y_pred=results, 
        num_classes=NUM_CLASSES, 
        ignore_background=True
    )

    # ==========================================
    # 4. 可视化结果 (可选：查看前3张图的对比)
    # ==========================================
    def plot_results(gen, predictions, num_samples=3):
        plt.figure(figsize=(15, 5 * num_samples))
        
        for i in range(num_samples):
            # 获取原始图和真值
            img_batch, mask_batch = gen[0] # 取第一个batch
            img = img_batch[i]
            gt = np.argmax(mask_batch[i], axis=-1)
            pred = np.argmax(predictions[i], axis=-1)
            
            # 绘制原始图
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f"Original CT {i}")
            plt.axis('off')
            
            # 绘制真值
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(gt, cmap='jet') # 使用jet彩色映射区分不同类
            plt.title(f"Ground Truth {i}")
            plt.axis('off')
            
            # 绘制预测值
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(pred, cmap='jet')
            plt.title(f"Prediction {i}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

    # 取消下面这行的注释即可看到对比图
    plot_results(val_gen, results)