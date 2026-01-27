import tensorflow as tf

import tensorflow as tf

def dice_loss(y_true, y_pred, smooth=1e-6, ignore_background=True):
    """
    多分类 Dice Loss
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # 【修复】如果模型最后已经有 softmax 了，这里不要再加 tf.nn.softmax
    # 如果担心数值稳定性，可以使用 clip
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)

    n = tf.shape(y_true)[0]
    c = tf.shape(y_true)[-1]

    # 将图片拉平进行计算
    y_true_f = tf.reshape(y_true, [n, -1, c])
    y_pred_f = tf.reshape(y_pred, [n, -1, c])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f + y_pred_f, axis=1)
    
    dice = (2. * intersection + smooth) / (denom + smooth)

    # 计算各类别平均 Dice
    dice_per_class = tf.reduce_mean(dice, axis=0) # 形状 [num_classes]

    if ignore_background is True:
        # 假设第 0 类是背景，只计算类别 1, 2, 3
        dice_per_class = dice_per_class[1:]

    loss = 1.0 - tf.reduce_mean(dice_per_class)
    return loss


def WCE_loss(weights):
    w = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        
        # 【修复】由于 y_pred 是概率值，不能用 softmax_cross_entropy_with_logits
        # 改用手动计算交叉熵或者调用 keras 的内置函数
        # 限制范围防止 log(0) 报错
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # 手动计算加权交叉熵: -sum(y_true * log(y_pred) * weights)
        log_post = tf.math.log(y_pred)
        weighted_ce = -tf.reduce_sum(y_true * log_post * w, axis=-1)
        
        return tf.reduce_mean(weighted_ce)
    
    return loss


def Combined_loss(weights, weight_of_dice, ignore_the_back=True):
    wce_loss_func = WCE_loss(weights)
    
    def loss(y_true, y_pred):
        d_loss = dice_loss(y_true, y_pred, ignore_background=ignore_the_back)
        w_loss = wce_loss_func(y_true, y_pred)
        return weight_of_dice * d_loss + (1 - weight_of_dice) * w_loss
        
    return loss