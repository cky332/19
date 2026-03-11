"""
GaussMarker 物理攻击鲁棒性测试
=================================
测试 GaussMarker 水印在以下物理攻击场景下的生存能力：

1. 打印-扫描攻击 (mild / moderate / heavy)
2. 屏幕翻拍攻击 (mild / moderate / heavy)
3. 透视变换攻击 (1° / 3° / 5° / 10° / 15°)
4. 组合攻击：
   - 打印扫描(moderate) + JPEG 压缩(quality=50)
   - 屏幕翻拍(moderate) + 缩放(0.5x)

使用方法:
    cd /path/to/19-main
    python test_physical_attacks.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import torch
import numpy as np
from PIL import Image
from io import BytesIO

from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from evaluation.tools.image_utils import (
    simulate_print_scan,
    simulate_screen_capture,
    simulate_perspective,
)


def fmt_result(result):
    """格式化检测结果的关键指标。"""
    return (
        f"is_watermarked={result['is_watermarked']:<6} "
        f"bit_acc={result['bit_acc']:.4f}  "
        f"fused_score={result['fused_score']:.4f}  "
        f"gnr_bit_acc={result['gnr_bit_acc']:.4f}"
    )


def jpeg_compress(image, quality):
    buf = BytesIO()
    image.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def main():
    # ==================================================================
    # 初始化：加载模型 & GaussMarker
    # ==================================================================
    print("=" * 80)
    print("GaussMarker 物理攻击鲁棒性测试")
    print("=" * 80)

    model_path = "huanzi05/stable-diffusion-2-1-base"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, scheduler=scheduler
    ).to(device)

    diffusion_config = DiffusionConfig(
        scheduler=scheduler,
        pipe=pipe,
        device=device,
        image_size=(512, 512),
        num_inference_steps=50,
        guidance_scale=3.5,
        gen_seed=42,
        inversion_type="ddim",
    )

    watermarker = AutoWatermark.load(
        'GM',
        algorithm_config='config/GM.json',
        diffusion_config=diffusion_config,
    )
    print("模型和 GaussMarker 加载完成!\n")

    # ==================================================================
    # 生成带水印图片
    # ==================================================================
    prompt = "A beautiful landscape with a river and mountains"
    print(f"Prompt: {prompt}")
    print("正在生成带水印图片...")
    watermarked_image = watermarker.generate_watermarked_media(
        input_data=prompt, guidance_scale=7.5
    )
    watermarked_image.save("output_watermarked.png")
    print("带水印图片已保存: output_watermarked.png\n")

    # 基准检测
    baseline = watermarker.detect_watermark_in_media(watermarked_image)
    print(f"[基准] 无攻击:  {fmt_result(baseline)}")
    print()

    # 收集所有测试结果
    results = []

    # ==================================================================
    # 测试 1：打印-扫描攻击
    # ==================================================================
    print("-" * 80)
    print("测试 1：打印-扫描攻击 (Print-Scan)")
    print("-" * 80)
    for severity in ['mild', 'moderate', 'heavy']:
        attacked = simulate_print_scan(watermarked_image, severity=severity)
        attacked.save(f"attacked_print_scan_{severity}.png")
        det = watermarker.detect_watermark_in_media(attacked)
        label = f"打印扫描({severity})"
        print(f"  [{label:<20}] {fmt_result(det)}")
        results.append((label, det))
    print()

    # ==================================================================
    # 测试 2：屏幕翻拍攻击
    # ==================================================================
    print("-" * 80)
    print("测试 2：屏幕翻拍攻击 (Screen Capture)")
    print("-" * 80)
    for severity in ['mild', 'moderate', 'heavy']:
        attacked = simulate_screen_capture(watermarked_image, severity=severity)
        attacked.save(f"attacked_screen_capture_{severity}.png")
        det = watermarker.detect_watermark_in_media(attacked)
        label = f"屏幕翻拍({severity})"
        print(f"  [{label:<20}] {fmt_result(det)}")
        results.append((label, det))
    print()

    # ==================================================================
    # 测试 3：透视变换攻击
    # ==================================================================
    print("-" * 80)
    print("测试 3：透视变换攻击 (Perspective)")
    print("-" * 80)
    for angle in [1, 3, 5, 10, 15]:
        attacked = simulate_perspective(watermarked_image,
                                        angle_x=angle, angle_y=angle)
        attacked.save(f"attacked_perspective_{angle}deg.png")
        det = watermarker.detect_watermark_in_media(attacked)
        label = f"透视变换({angle}°)"
        print(f"  [{label:<20}] {fmt_result(det)}")
        results.append((label, det))
    print()

    # ==================================================================
    # 测试 4：组合攻击
    # ==================================================================
    print("-" * 80)
    print("测试 4：组合攻击 (Combined)")
    print("-" * 80)

    # 4a. 打印扫描(moderate) + JPEG 压缩(quality=50)
    attacked = simulate_print_scan(watermarked_image, severity='moderate')
    attacked = jpeg_compress(attacked, quality=50)
    attacked.save("attacked_combined_ps_jpeg.png")
    det = watermarker.detect_watermark_in_media(attacked)
    label = "打印扫描+JPEG50"
    print(f"  [{label:<20}] {fmt_result(det)}")
    results.append((label, det))

    # 4b. 屏幕翻拍(moderate) + 缩放(0.5x)
    attacked = simulate_screen_capture(watermarked_image, severity='moderate')
    w, h = attacked.size
    attacked = attacked.resize((w // 2, h // 2), Image.BILINEAR)
    attacked = attacked.resize((w, h), Image.BILINEAR)
    attacked.save("attacked_combined_sc_resize.png")
    det = watermarker.detect_watermark_in_media(attacked)
    label = "屏幕翻拍+缩放0.5x"
    print(f"  [{label:<20}] {fmt_result(det)}")
    results.append((label, det))
    print()

    # ==================================================================
    # 汇总报告
    # ==================================================================
    print("=" * 80)
    print("汇总报告")
    print("=" * 80)
    print(f"{'攻击方式':<22} {'水印存活':>8} {'bit_acc':>10} {'fused_score':>12} {'gnr_bit_acc':>12}")
    print("-" * 70)
    print(f"{'无攻击(基准)':<22} {'✓':>8} {baseline['bit_acc']:>10.4f} {baseline['fused_score']:>12.4f} {baseline['gnr_bit_acc']:>12.4f}")
    for label, det in results:
        survived = "✓" if det['is_watermarked'] else "✗"
        print(f"{label:<22} {survived:>8} {det['bit_acc']:>10.4f} {det['fused_score']:>12.4f} {det['gnr_bit_acc']:>12.4f}")

    total = len(results)
    survived_count = sum(1 for _, det in results if det['is_watermarked'])
    print("-" * 70)
    print(f"水印存活率: {survived_count}/{total} ({survived_count/total*100:.1f}%)")
    print("=" * 80)


if __name__ == '__main__':
    main()
