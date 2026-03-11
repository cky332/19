"""
GaussMarker 完整流程验证脚本
==============================
流程: 加载模型 → 嵌入水印生成图片 → 检测水印 → 生成无水印图片对比

使用方法:
    cd /path/to/19-main
    python verify_gaussmarker.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ============================================================================
# 第一步：加载 Stable Diffusion 模型，构建 DiffusionConfig
# ============================================================================
print("=" * 60)
print("[1/5] 加载 Stable Diffusion 2.1 模型...")
print("=" * 60)

model_path = "stabilityai/stable-diffusion-2-1-base"

scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
pipe = pipe.to(device)

diffusion_config = DiffusionConfig(
    scheduler=scheduler,
    pipe=pipe,
    device=device,
    image_size=(512, 512),
    num_inference_steps=50,
    guidance_scale=3.5,
    gen_seed=42,
    inversion_type="ddim"
)
print("模型加载完成!\n")

# ============================================================================
# 第二步：加载 GaussMarker 水印算法
# ============================================================================
print("=" * 60)
print("[2/5] 加载 GaussMarker (GM) 水印算法...")
print("=" * 60)

watermarker = AutoWatermark.load(
    'GM',
    algorithm_config='config/GM.json',
    diffusion_config=diffusion_config
)
print("GaussMarker 加载完成!\n")

# ============================================================================
# 第三步：生成带水印的图片
# ============================================================================
print("=" * 60)
print("[3/5] 生成带水印的图片...")
print("=" * 60)

prompt = "A beautiful landscape with a river and mountains"
watermarked_image = watermarker.generate_watermarked_media(
    input_data=prompt,
    guidance_scale=7.5
)

watermarked_image.save("output_watermarked.png")
print(f"带水印图片已保存: output_watermarked.png\n")

# ============================================================================
# 第四步：检测水印
# ============================================================================
print("=" * 60)
print("[4/5] 检测水印...")
print("=" * 60)

# 检测带水印图片
print("--- 检测带水印图片 ---")
result_wm = watermarker.detect_watermark_in_media(watermarked_image)
print(f"检测结果: {result_wm}")
print()

# 生成无水印图片并检测（对比）
print("--- 生成并检测无水印图片（对比） ---")
unwatermarked_image = watermarker.generate_unwatermarked_media(input_data=prompt)
unwatermarked_image.save("output_unwatermarked.png")

result_no_wm = watermarker.detect_watermark_in_media(unwatermarked_image)
print(f"检测结果: {result_no_wm}")
print()

# ============================================================================
# 第五步：结果汇总
# ============================================================================
print("=" * 60)
print("[5/5] 验证结果汇总")
print("=" * 60)
print(f"Prompt:           {prompt}")
print(f"带水印图片:       output_watermarked.png")
print(f"无水印图片:       output_unwatermarked.png")
print(f"带水印检测结果:   {result_wm}")
print(f"无水印检测结果:   {result_no_wm}")
print()
print("如果带水印图片被检测为含水印、无水印图片未检测到水印，")
print("则说明 GaussMarker 完整流程验证成功!")
print("=" * 60)
