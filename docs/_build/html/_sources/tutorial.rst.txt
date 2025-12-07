Tutorial
========

This tutorial provides step-by-step examples for using MarkDiffusion's main features.

Tutorial 1: Basic Image Watermarking
-------------------------------------

Learn how to watermark images using different algorithms.

Using Tree-Ring Watermark
~~~~~~~~~~~~~~~~~~~~~~~~~~

Tree-Ring is a pattern-based watermarking method that embeds invisible patterns in the frequency domain.

.. code-block:: python

   import torch
   from watermark.auto_watermark import AutoWatermark
   from utils.diffusion_config import DiffusionConfig
   from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

   # Setup
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model_id = "stabilityai/stable-diffusion-2-1"
   
   scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
   pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler).to(device)
   
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
   
   # Load Tree-Ring watermark
   tr_watermark = AutoWatermark.load(
       'TR',
       algorithm_config='config/TR.json',
       diffusion_config=diffusion_config
   )
   
   # Generate watermarked image
   prompt = "A majestic mountain landscape with snow peaks"
   watermarked_img = tr_watermark.generate_watermarked_media(prompt)
   watermarked_img.save("tr_watermarked.png")
   
   # Detect watermark
   result = tr_watermark.detect_watermark_in_media(watermarked_img)
   print(f"Tree-Ring Detection: {result}")

Using Gaussian-Shading Watermark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gaussian-Shading is a key-based method that provides provable performance-lossless watermarking.

.. code-block:: python

   # Load Gaussian-Shading watermark
   gs_watermark = AutoWatermark.load(
       'GS',
       algorithm_config='config/GS.json',
       diffusion_config=diffusion_config
   )
   
   # Generate watermarked image
   watermarked_img = gs_watermark.generate_watermarked_media(prompt)
   watermarked_img.save("gs_watermarked.png")
   
   # Detect watermark
   result = gs_watermark.detect_watermark_in_media(watermarked_img)
   print(f"Gaussian-Shading Detection: {result}")

Tutorial 2: Visualizing Watermark Mechanisms
---------------------------------------------

Learn how to visualize watermarking mechanisms using the same approach as in ``MarkDiffusion_demo.ipynb``.

Tree-Ring Visualization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from visualize.auto_visualization import AutoVisualizer

   # Get visualization data
   data_for_visualization = mywatermark.get_data_for_visualize(watermarked_image)

   # Load visualizer
   visualizer = AutoVisualizer.load('TR', data_for_visualization=data_for_visualization)

   # Create visualization with specific methods and channels
   method_kwargs = [{}, {"channel": 0}, {}, {"channel": 0}, {}]
   fig = visualizer.visualize(
       rows=1,
       cols=5,
       methods=['draw_pattern_fft', 'draw_orig_latents_fft', 'draw_watermarked_image', 
                'draw_inverted_latents_fft', 'draw_inverted_pattern_fft'],
       method_kwargs=method_kwargs,
       save_path='TR_watermark_visualization.pdf'
   )

Gaussian-Shading Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from visualize.auto_visualization import AutoVisualizer

   data_for_visualization = mywatermark.get_data_for_visualize(watermarked_image)
   visualizer = AutoVisualizer.load('GS', data_for_visualization=data_for_visualization)

   method_kwargs = [{"channel": 0}, {"channel": 0}, {}, {"channel": 0}, {"channel": 0}]
   fig = visualizer.visualize(
       rows=1, 
       cols=5, 
       methods=['draw_watermark_bits', 'draw_orig_latents', 'draw_watermarked_image', 
                'draw_inverted_latents', 'draw_reconstructed_watermark_bits'], 
       method_kwargs=method_kwargs, 
       save_path='GS_watermark_visualization.pdf'
   )

ROBIN Visualization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from visualize.auto_visualization import AutoVisualizer

   data_for_visualization = mywatermark.get_data_for_visualize(watermarked_image)
   visualizer = AutoVisualizer.load('ROBIN', data_for_visualization=data_for_visualization)

   method_kwargs = [{}, {"channel": 3}, {}, {"channel": 3}, {}]
   fig = visualizer.visualize(
       rows=1, 
       cols=5, 
       methods=['draw_pattern_fft', 'draw_orig_latents_fft', 'draw_watermarked_image', 
                'draw_inverted_latents_fft', 'draw_inverted_pattern_fft'], 
       method_kwargs=method_kwargs, 
       save_path='ROBIN_watermark_visualization.pdf'
   )

Tutorial 3: Evaluating Watermark Quality
-----------------------------------------

Assess the quality and robustness of watermarks.

Image Quality Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from evaluation.dataset import StableDiffusionPromptsDataset
   from evaluation.pipelines.image_quality_analysis import ComparedImageQualityAnalysisPipeline
   from evaluation.tools.image_quality_analyzer import PSNRAnalyzer, SSIMAnalyzer
   from evaluation.pipelines.image_quality_analysis import QualityPipelineReturnType

   # Create dataset
   dataset = StableDiffusionPromptsDataset(max_samples=50)
   
   # Setup quality analysis pipeline
   pipeline = ComparedImageQualityAnalysisPipeline(
       dataset=dataset,
       watermarked_image_editor_list=[],
       unwatermarked_image_editor_list=[],
       analyzers=[PSNRAnalyzer(), SSIMAnalyzer()],
       show_progress=True,
       return_type=QualityPipelineReturnType.MEAN_SCORES
   )
   
   # Evaluate watermark quality
   results = pipeline.evaluate(gs_watermark)
   print(f"PSNR: {results['PSNR']:.2f} dB")
   print(f"SSIM: {results['SSIM']:.4f}")

Robustness Evaluation
~~~~~~~~~~~~~~~~~~~~~~

Test watermark robustness against various attacks:

.. code-block:: python

   from evaluation.tools.image_editor import (
       JPEGCompression, GaussianBlurring, GaussianNoise, Rotation
   )
   from evaluation.pipelines.detection import WatermarkedMediaDetectionPipeline
   from evaluation.pipelines.detection import DetectionPipelineReturnType
   from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator

   # Test against JPEG compression
   dataset = StableDiffusionPromptsDataset(max_samples=100)
   
   # Create detection pipeline with JPEG attack
   detection_pipeline = WatermarkedMediaDetectionPipeline(
       dataset=dataset,
       media_editor_list=[JPEGCompression(quality=75)],
       show_progress=True,
       return_type=DetectionPipelineReturnType.SCORES
   )
   
   # Evaluate detection after attack
   detection_kwargs = {
       "num_inference_steps": 50,
       "guidance_scale": 1.0,
   }
   
   scores = detection_pipeline.evaluate(gs_watermark, detection_kwargs=detection_kwargs)
   print(f"Detection scores after JPEG compression: {scores}")

Multiple Attacks Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test robustness against multiple attacks
   attacks = {
       'JPEG-75': JPEGCompression(quality=75),
       'JPEG-50': JPEGCompression(quality=50),
       'Blur': GaussianBlurring(kernel_size=3),
       'Noise': GaussianNoise(std=0.05),
       'Rotation': Rotation(angle=15)
   }
   
   results = {}
   for attack_name, attack_editor in attacks.items():
       pipeline = WatermarkedMediaDetectionPipeline(
           dataset=dataset,
           media_editor_list=[attack_editor],
           show_progress=True,
           return_type=DetectionPipelineReturnType.SCORES
       )
       scores = pipeline.evaluate(gs_watermark, detection_kwargs=detection_kwargs)
       results[attack_name] = scores
   
   # Print results
   print("\n=== Robustness Results ===")
   for attack, scores in results.items():
       avg_score = sum(scores) / len(scores) if scores else 0
       print(f"{attack}: Average Score = {avg_score:.4f}")

Tutorial 4: Video Watermarking
-------------------------------

Learn how to watermark videos using VideoShield or VideoMark.

Basic Video Watermarking
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from diffusers import DiffusionPipeline
   
   # Setup for video generation
   video_model_id = "cerspense/zeroscope_v2_576w"
   video_pipe = DiffusionPipeline.from_pretrained(
       video_model_id,
       torch_dtype=torch.float16
   ).to(device)
   
   video_diffusion_config = DiffusionConfig(
       scheduler=video_pipe.scheduler,
       pipe=video_pipe,
       device=device,
       image_size=(576, 320),
       num_inference_steps=40,
       guidance_scale=7.5,
       gen_seed=42,
       num_frames=16
   )
   
   # Load VideoShield watermark
   video_watermark = AutoWatermark.load(
       'VideoShield',
       algorithm_config='config/VideoShield.json',
       diffusion_config=video_diffusion_config
   )
   
   # Generate watermarked video
   prompt = "A drone flying over a beach at sunset"
   video_frames = video_watermark.generate_watermarked_media(prompt)
   
   # Save frames
   import os
   os.makedirs("output_video", exist_ok=True)
   for i, frame in enumerate(video_frames):
       frame.save(f"output_video/frame_{i:04d}.png")
   
   # Detect watermark in video
   detection_result = video_watermark.detect_watermark_in_media(video_frames)
   print(f"Video watermark detection: {detection_result}")

Video Quality Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from evaluation.dataset import VBenchDataset
   from evaluation.pipelines.video_quality_analysis import DirectVideoQualityAnalysisPipeline
   from evaluation.tools.video_quality_analyzer import (
       SubjectConsistencyAnalyzer,
       MotionSmoothnessAnalyzer
   )
   
   # Create video dataset
   video_dataset = VBenchDataset(max_samples=20, dimension='subject_consistency')
   
   # Setup video quality pipeline
   video_pipeline = DirectVideoQualityAnalysisPipeline(
       dataset=video_dataset,
       watermarked_video_editor_list=[],
       unwatermarked_video_editor_list=[],
       watermarked_frame_editor_list=[],
       unwatermarked_frame_editor_list=[],
       analyzers=[SubjectConsistencyAnalyzer(device=device)],
       show_progress=True,
       return_type=QualityPipelineReturnType.MEAN_SCORES
   )
   
   # Evaluate video quality
   results = video_pipeline.evaluate(video_watermark)
   print(f"Subject consistency: {results}")

Tutorial 5: Custom Configuration
---------------------------------

Customize watermarking parameters for your specific needs.

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   prompts = [
       "A serene lake at dawn",
       "A bustling city street at night",
       "A colorful flower garden",
       "A snowy mountain peak",
       "A tropical beach paradise"
   ]
   
   output_dir = "batch_output"
   os.makedirs(output_dir, exist_ok=True)
   
   for i, prompt in enumerate(prompts):
       print(f"Processing {i+1}/{len(prompts)}: {prompt}")
       
       # Generate watermarked image
       img = gs_watermark.generate_watermarked_media(prompt)
       img.save(f"{output_dir}/watermarked_{i:03d}.png")
       
       # Detect watermark
       result = gs_watermark.detect_watermark_in_media(img)
       print(f"  Detection: {result}")

Next Steps
----------

Explore more advanced topics:

- :doc:`user_guide/algorithms` - Detailed algorithm documentation
- :doc:`api/watermark` - Complete API reference

