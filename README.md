# 2D-Chest-CT-Seg

**概况**：该项目用于处理2D CT胸部器官分割任务，需要使用图像和目标掩码从0开始进行训练
1. 数据类型：输入为256*256灰度图像，masks通过内置的转换程序由rgb彩色图像转换为Label Map（标签图）
2. 内含unet,simple_nestnet,nestnet with backbone三种模型选择

下面是经过训练后模型分割的实例：

