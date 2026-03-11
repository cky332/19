"""
物理攻击模拟工具
==================
模拟三种常见物理攻击场景：
1. 打印-扫描攻击 (simulate_print_scan)
2. 屏幕翻拍攻击 (simulate_screen_capture)
3. 透视变换攻击 (simulate_perspective)
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO
import math


def _solve_perspective_coeffs(src_points, dst_points):
    """
    求解 8 参数透视变换矩阵。
    通过 4 对点的对应关系求解线性方程组 Ax = b。
    """
    matrix = []
    for s, d in zip(src_points, dst_points):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0]*d[0], -s[0]*d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1]*d[0], -s[1]*d[1]])
    A = np.array(matrix, dtype=np.float64)
    b = np.array([p for s in src_points for p in s], dtype=np.float64)
    coeffs = np.linalg.solve(A, b)
    return tuple(coeffs.tolist())


def _jpeg_compress(image, quality):
    """内存中的 JPEG 压缩（避免临时文件）。"""
    buf = BytesIO()
    image.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def simulate_perspective(image, angle_x=0.0, angle_y=0.0, focal=1000.0):
    """
    透视变换攻击：模拟从非正面角度拍摄/观看图像导致的几何畸变。

    原理：
    1. 将图像四角视为三维空间中的点
    2. 绕 X 轴旋转 angle_x（上下倾斜），绕 Y 轴旋转 angle_y（左右倾斜）
    3. 通过透视投影计算新角点位置
    4. 求解 8 参数透视变换矩阵并应用

    Args:
        image: PIL Image
        angle_x: 绕 X 轴旋转角度（度）
        angle_y: 绕 Y 轴旋转角度（度）
        focal: 焦距参数（越大透视效果越弱）
    """
    w, h = image.size
    ax = math.radians(angle_x)
    ay = math.radians(angle_y)

    # 以图像中心为原点的四角坐标
    corners = [
        (-w/2, -h/2),  # TL
        ( w/2, -h/2),  # TR
        ( w/2,  h/2),  # BR
        (-w/2,  h/2),  # BL
    ]

    dst_corners = []
    for cx, cy in corners:
        # 绕 Y 轴旋转后透视投影
        xr = cx * math.cos(ay) / (1 - cx * math.sin(ay) / focal)
        # 绕 X 轴旋转后透视投影
        yr = cy * math.cos(ax) / (1 - cy * math.sin(ax) / focal)
        dst_corners.append((xr + w/2, yr + h/2))

    src_points = [(0, 0), (w, 0), (w, h), (0, h)]
    coeffs = _solve_perspective_coeffs(src_points, dst_corners)
    return image.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def simulate_print_scan(image, severity='moderate'):
    """
    打印-扫描攻击：模拟将图像打印到纸上再用扫描仪扫描回电子版的过程。

    攻击流水线（7 步）：
    1. 色彩深度降低 - 打印机墨水无法还原全部色彩
    2. DPI 分辨率损失 - 打印/扫描的分辨率有限
    3. 高斯模糊 - 墨水在纸上的扩散
    4. 亮度/对比度偏移 - 打印油墨与屏幕显色差异
    5. 高斯噪声 - 扫描仪传感器噪声
    6. 轻微透视变换 - 纸张放置不平整
    7. JPEG 压缩 - 扫描仪输出格式

    Args:
        image: PIL Image
        severity: 'mild', 'moderate', 'heavy'
    """
    presets = {
        'mild':     dict(depth=7, blur=0.5, noise=0.01, jpeg=85, dpi=0.85,
                         brightness=0.95, contrast=0.95, angle=0.5),
        'moderate': dict(depth=6, blur=1.0, noise=0.02, jpeg=70, dpi=0.7,
                         brightness=0.90, contrast=0.90, angle=1.0),
        'heavy':    dict(depth=5, blur=2.0, noise=0.04, jpeg=50, dpi=0.5,
                         brightness=0.85, contrast=0.80, angle=2.0),
    }
    p = presets[severity]
    w, h = image.size

    # 1. 色彩深度降低
    arr = np.array(image)
    shift = 8 - p['depth']
    arr = (arr >> shift) << shift
    img = Image.fromarray(arr)

    # 2. DPI 分辨率损失：缩小再放大
    small_size = (int(w * p['dpi']), int(h * p['dpi']))
    img = img.resize(small_size, Image.BILINEAR).resize((w, h), Image.BILINEAR)

    # 3. 高斯模糊
    img = img.filter(ImageFilter.GaussianBlur(radius=p['blur']))

    # 4. 亮度/对比度偏移
    img = ImageEnhance.Brightness(img).enhance(p['brightness'])
    img = ImageEnhance.Contrast(img).enhance(p['contrast'])

    # 5. 高斯噪声
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, p['noise'] * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # 6. 轻微透视变换
    img = simulate_perspective(img, angle_x=p['angle'], angle_y=p['angle'])

    # 7. JPEG 压缩
    img = _jpeg_compress(img, p['jpeg'])

    return img


def simulate_screen_capture(image, severity='moderate'):
    """
    屏幕翻拍攻击：模拟在显示器上显示图片后用相机拍摄屏幕的过程。

    攻击流水线（8 步）：
    1. Gamma 失配 - 显示器 gamma 与相机 gamma 不匹配
    2. 色温偏移 - 显示器色温与环境光差异
    3. 摩尔纹 - 像素网格与相机传感器干涉（仅 heavy）
    4. 高斯模糊 - 相机对焦不准
    5. 分辨率损失 - 相机分辨率低于原图
    6. 高斯噪声 - 相机传感器噪声
    7. 透视变换 - 拍摄角度不正
    8. JPEG 压缩 - 相机保存格式

    Args:
        image: PIL Image
        severity: 'mild', 'moderate', 'heavy'
    """
    presets = {
        'mild':     dict(gamma=1.1, blur=0.3, noise=0.01, angle=0.5,
                         scale=0.9, jpeg=90, moire=False,
                         color_shift=(5, -3, -5)),
        'moderate': dict(gamma=1.3, blur=0.8, noise=0.025, angle=2.0,
                         scale=0.75, jpeg=75, moire=False,
                         color_shift=(10, -5, -10)),
        'heavy':    dict(gamma=1.6, blur=1.5, noise=0.05, angle=5.0,
                         scale=0.5, jpeg=55, moire=True,
                         color_shift=(20, -8, -15)),
    }
    p = presets[severity]
    w, h = image.size

    # 1. Gamma 失配
    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.power(arr, p['gamma'])
    arr = (arr * 255).clip(0, 255).astype(np.uint8)

    # 2. 色温偏移
    r_shift, g_shift, b_shift = p['color_shift']
    arr[:, :, 0] = np.clip(arr[:, :, 0].astype(np.int16) + r_shift, 0, 255).astype(np.uint8)
    arr[:, :, 1] = np.clip(arr[:, :, 1].astype(np.int16) + g_shift, 0, 255).astype(np.uint8)
    arr[:, :, 2] = np.clip(arr[:, :, 2].astype(np.int16) + b_shift, 0, 255).astype(np.uint8)

    # 3. 摩尔纹（仅 heavy）
    if p['moire']:
        freq = 50
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        moire_pattern = (np.sin(2 * np.pi * freq * xx / w) *
                         np.sin(2 * np.pi * freq * yy / h) * 10)
        moire_pattern = np.stack([moire_pattern] * 3, axis=-1)
        arr = np.clip(arr.astype(np.float32) + moire_pattern, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)

    # 4. 高斯模糊
    img = img.filter(ImageFilter.GaussianBlur(radius=p['blur']))

    # 5. 分辨率损失
    small_size = (int(w * p['scale']), int(h * p['scale']))
    img = img.resize(small_size, Image.BILINEAR).resize((w, h), Image.BILINEAR)

    # 6. 高斯噪声
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, p['noise'] * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # 7. 透视变换
    img = simulate_perspective(img, angle_x=p['angle'], angle_y=p['angle'] * 0.5)

    # 8. JPEG 压缩
    img = _jpeg_compress(img, p['jpeg'])

    return img
