import tensorflow as tf

import tensorflow as tf

def multiclass_dice(y_true, y_pred, smooth=1e-6, ignore_background=True):
    """
    用于评估的硬性 Dice 系数 (Hard Dice)
    """
    # 1. 确保数据类型
    y_true = tf.cast(y_true, tf.float32)
    
    # 2. 【核心修改】将预测概率转换为类别索引，再转回 one-hot
    # 这样计算出来的才是真实的像素重合率
    y_pred_idx = tf.argmax(y_pred, axis=-1) 
    y_pred_onehot = tf.one_hot(y_pred_idx, depth=tf.shape(y_true)[-1])
    y_pred_onehot = tf.cast(y_pred_onehot, tf.float32)

    # 3. 展平 (Flatten)
    # 形状从 [batch, h, w, c] 变为 [batch, pixels, c]
    n = tf.shape(y_true)[0]
    c = tf.shape(y_true)[-1]
    y_true_f = tf.reshape(y_true, [n, -1, c])
    y_pred_f = tf.reshape(y_pred_onehot, [n, -1, c])

    # 4. 计算交集和并集
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f + y_pred_f, axis=1)
    
    # 计算每个 Batch 每个类别的 Dice
    dice = (2. * intersection + smooth) / (denom + smooth)

    # 5. 计算所有样本的平均值
    dice_per_class = tf.reduce_mean(dice, axis=0)

    # 6. 是否忽略背景（通常背景是第0类）
    if ignore_background:
        # 只取 1, 2, 3 类
        dice_per_class = dice_per_class[1:]

    # 返回所有类别的平均 Dice (Mean Dice)
    return tf.reduce_mean(dice_per_class)