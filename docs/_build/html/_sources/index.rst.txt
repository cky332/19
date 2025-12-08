MarkDiffusion Documentation
============================

.. image:: https://img.shields.io/badge/Homepage-5F259F?style=for-the-badge&logo=homepage&logoColor=white
   :target: https://generative-watermark.github.io/
   :alt: Homepage

.. image:: https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white
   :target: https://arxiv.org/abs/2509.10569
   :alt: Paper

.. image:: https://img.shields.io/badge/HF--Models-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black
   :target: https://huggingface.co/Generative-Watermark-Toolkits
   :alt: HF Models

Welcome to MarkDiffusion
-------------------------

**MarkDiffusion** is an open-source Python toolkit for generative watermarking of latent diffusion models. 
As the use of diffusion-based generative models expands, ensuring the authenticity and origin of generated 
media becomes critical. MarkDiffusion simplifies the access, understanding, and assessment of watermarking 
technologies, making it accessible to both researchers and the broader community.

.. note::
   If you are interested in LLM watermarking (text watermark), please refer to the 
   `MarkLLM <https://github.com/THU-BPM/MarkLLM>`_ toolkit from our group.

Key Features
------------

🚀 **Unified Implementation Framework**
   MarkDiffusion provides a modular architecture supporting eleven state-of-the-art generative 
   image/video watermarking algorithms of LDMs.

📦 **Comprehensive Algorithm Support**
   Currently implements 11 watermarking algorithms from two major categories:
   
   - **Pattern-based methods**: Tree-Ring, Ring-ID, ROBIN, WIND, SFW
   - **Key-based methods**: Gaussian-Shading, GaussMarker, PRC, SEAL, VideoShield, VideoMark

🔍 **Visualization Solutions**
   The toolkit includes custom visualization tools that enable clear and insightful views into 
   how different watermarking algorithms operate under various scenarios.

📊 **Comprehensive Evaluation Module**
   With 24 evaluation tools covering detectability, robustness, and impact on output quality, 
   MarkDiffusion provides comprehensive assessment capabilities with 8 automated evaluation pipelines.

Quick Example
-------------

Here's a simple example to get you started with MarkDiffusion:

.. code-block:: python

    import torch
    from watermark.auto_watermark import AutoWatermark
    from utils.diffusion_config import DiffusionConfig
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configure diffusion pipeline
    scheduler = DPMSolverMultistepScheduler.from_pretrained("model_path", subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained("model_path", scheduler=scheduler).to(device)
    diffusion_config = DiffusionConfig(
        scheduler=scheduler,
        pipe=pipe,
        device=device,
        image_size=(512, 512),
        num_inference_steps=50,
        guidance_scale=7.5,
        gen_seed=42,
        inversion_type="ddim"
    )

    # Load watermark algorithm
    watermark = AutoWatermark.load('TR', 
                                  algorithm_config='config/TR.json',
                                  diffusion_config=diffusion_config)

    # Generate watermarked media
    prompt = "A beautiful sunset over the ocean"
    watermarked_image = watermark.generate_watermarked_media(prompt)

    # Detect watermark
    detection_result = watermark.detect_watermark_in_media(watermarked_image)
    print(f"Watermark detected: {detection_result}")

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/algorithms
   user_guide/watermarking
   user_guide/visualization
   user_guide/evaluation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/watermark
   api/detection
   api/visualization
   api/evaluation
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   contributing
   code_of_conduct
   citation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

