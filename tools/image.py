import numpy as np
from copy import copy


def draw_msra_gaussian_3d(heatmap_3d, center, sigma):
    """
    在三维热图上以指定中心点和标准差绘制三维高斯分布, 并与原始热图融合(取最大值), 返回更新后的三维热图。

    :param heatmap_3d: 三维热图, 形状为 (depth, height, width)
    :param center: 中心点坐标, 格式为 (x, y, z)
    :param sigma: 高斯分布的标准差
    :return: 更新后的三维热图
    """

    # 确定临时尺寸
    tmp_size = sigma * 3

    # 将中心点坐标转换为整数形式(四舍五入)
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    mu_z = int(center[2] + 0.5)

    # 获取三维热图的尺寸
    w, h, d = heatmap_3d.shape

    # 确定绘制区域的边界坐标
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size), int(mu_z - tmp_size)] # 下界(左下)
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1), int(mu_z + tmp_size + 1)] # 上界(右上)

    # 如果绘制区域不在热图上则返回原始热图
    if (ul[0] >= w or ul[1] >= h or ul[2] >= d or br[0] < 0 or br[1] < 0 or br[2] < 0):
        return heatmap_3d

    # 确定生成高斯分布数据的尺寸
    size = 2 * tmp_size + 1

    # 生成三维坐标网格
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    z = y[:, :, np.newaxis]

    # 确定高斯分布的中心点坐标
    x0 = y0 = z0 = size // 2
    
    # 计算三维高斯分布数据
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma ** 2))
    
    # 确定高斯分布数据和热图的有效区域索引
    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    g_z = max(0, -ul[2]), min(br[2], d) - ul[2]

    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)
    img_z = max(0, ul[2]), min(br[2], d)

    # 将三维高斯分布与热图融合(取最大值操作)
    heatmap_3d[img_x[0]:img_x[1], img_y[0]:img_y[1], img_z[0]:img_z[1]] = np.maximum(
            heatmap_3d[img_x[0]:img_x[1], img_y[0]:img_y[1], img_z[0]:img_z[1]], 
            g[g_x[0]:g_x[1], g_y[0]:g_y[1], g_z[0]:g_z[1]]
        )

    return heatmap_3d


def crop_img_from_keypoint(img, point, crop_size):
    Z,Y,X = img.shape # zyx
    temp_size = [i//2 for i in crop_size]

    mu_z = int(point[0] + 0.5)
    mu_y = int(point[1] + 0.5)
    mu_x = int(point[2] + 0.5)
    
    new_point = [mu_z,mu_y, mu_x]
    
    if new_point[0] + temp_size[0] > Z:
        new_point[0] = Z - temp_size[0]
    if new_point[0] - temp_size[0] < 0:
        new_point[0] = temp_size[0]
    if new_point[1] + temp_size[1] > Y:
        new_point[1] = Y - temp_size[1]
    if new_point[1] - temp_size[1] < 0:
        new_point[1] = temp_size[1]
    if new_point[2] + temp_size[2] > X:
        new_point[2] = X - temp_size[2]
    if new_point[2] - temp_size[2] < 0:
        new_point[2] = temp_size[2]
    
    return copy(
        img[new_point[0]-temp_size[0]:new_point[0]+temp_size[0],
            new_point[1]-temp_size[1]:new_point[1]+temp_size[1],
            new_point[2]-temp_size[2]:new_point[2]+temp_size[2]])