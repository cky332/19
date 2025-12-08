Quick Start
===========

This guide will help you get started with MarkDiffusion quickly.

Basic Workflow
--------------

The typical workflow with MarkDiffusion consists of three main steps:

1. **Configure** the diffusion model and watermarking algorithm
2. **Generate** watermarked images or videos
3. **Detect** watermarks in media

Step 1: Setup and Configuration
--------------------------------

First, import the necessary modules and configure your diffusion model:

.. code-block:: python

   import torch
   from watermark.auto_watermark import AutoWatermark
   from utils.diffusion_config import DiffusionConfig
   from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

   # Device setup
   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   # Configure the diffusion model
   model_id = "stabilityai/stable-diffusion-2-1"
   scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
   pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler).to(device)

   # Create diffusion configuration
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

Step 2: Load a Watermarking Algorithm
--------------------------------------

MarkDiffusion supports multiple watermarking algorithms. Here's how to load one:

.. code-block:: python

   # Load Tree-Ring watermarking algorithm
   watermark = AutoWatermark.load(
       'TR',                              # Algorithm name
       algorithm_config='config/TR.json', # Configuration file
       diffusion_config=diffusion_config  # Diffusion settings
   )

Available algorithms:

- **TR**: Tree-Ring
- **RI**: Ring-ID
- **ROBIN**: ROBIN
- **WIND**: WIND
- **SFW**: Semantic Fourier Watermark
- **GS**: Gaussian-Shading
- **GM**: GaussMarker
- **PRC**: PRC
- **SEAL**: SEAL
- **VideoShield**: VideoShield
- **VideoMark**: VideoMark

Step 3: Generate Watermarked Media
-----------------------------------

Image Generation
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define your prompt
   prompt = "A beautiful landscape with mountains and a lake at sunset"

   # Generate watermarked image
   watermarked_image = watermark.generate_watermarked_media(prompt)

   # Save the image
   watermarked_image.save("watermarked_output.png")

   # Display the image
   watermarked_image.show()

Video Generation
~~~~~~~~~~~~~~~~

For video watermarking algorithms (VideoShield, VideoMark):

.. code-block:: python

   # Load video watermarking algorithm
   video_watermark = AutoWatermark.load(
       'VideoShield',
       algorithm_config='config/VideoShield.json',
       diffusion_config=video_diffusion_config
   )

   # Generate watermarked video
   prompt = "A cat walking through a garden"
   video_frames = video_watermark.generate_watermarked_media(prompt)

   # Save video frames
   for i, frame in enumerate(video_frames):
       frame.save(f"output/frame_{i:04d}.png")

Step 4: Detect Watermarks
--------------------------

After generating watermarked media, you can detect the watermark:

.. code-block:: python

   # Detect watermark in the generated image
   detection_result = watermark.detect_watermark_in_media(watermarked_image)

   # Print detection results
   print(f"Detection result: {detection_result}")

   # For algorithms that return a score
   if 'score' in detection_result:
       print(f"Confidence score: {detection_result['score']}")
       print(f"Threshold: {detection_result.get('threshold', 'N/A')}")

Complete Example
----------------

Here's a complete example putting it all together:

.. code-block:: python

   import torch
   from watermark.auto_watermark import AutoWatermark
   from utils.diffusion_config import DiffusionConfig
   from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

   def main():
       # Setup
       device = 'cuda' if torch.cuda.is_available() else 'cpu'
       model_id = "stabilityai/stable-diffusion-2-1"
       
       # Load diffusion model
       scheduler = DPMSolverMultistepScheduler.from_pretrained(
           model_id, subfolder="scheduler"
       )
       pipe = StableDiffusionPipeline.from_pretrained(
           model_id, scheduler=scheduler
       ).to(device)
       
       # Configure diffusion
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
       
       # Load watermarking algorithm
       watermark = AutoWatermark.load(
           'GS',  # Gaussian-Shading
           algorithm_config='config/GS.json',
           diffusion_config=diffusion_config
       )
       
       # Generate watermarked image
       prompt = "A serene Japanese garden with cherry blossoms"
       print("Generating watermarked image...")
       watermarked_image = watermark.generate_watermarked_media(prompt)
       
       # Save image
       watermarked_image.save("output.png")
       print("Image saved to output.png")
       
       # Detect watermark
       print("Detecting watermark...")
       detection_result = watermark.detect_watermark_in_media(watermarked_image)
       print(f"Detection result: {detection_result}")
       
       return watermarked_image, detection_result

   if __name__ == "__main__":
       image, result = main()

Comparing Multiple Algorithms
------------------------------

You can easily compare different algorithms:

.. code-block:: python

   algorithms = ['TR', 'GS', 'ROBIN', 'SEAL']
   prompt = "A futuristic city skyline at night"

   results = {}
   for algo_name in algorithms:
       print(f"\nTesting {algo_name}...")
       
       # Load algorithm
       watermark = AutoWatermark.load(
           algo_name,
           algorithm_config=f'config/{algo_name}.json',
           diffusion_config=diffusion_config
       )
       
       # Generate and detect
       img = watermark.generate_watermarked_media(prompt)
       detection = watermark.detect_watermark_in_media(img)
       
       results[algo_name] = {
           'image': img,
           'detection': detection
       }
       
       # Save image
       img.save(f"output_{algo_name}.png")

   # Print comparison
   print("\n=== Comparison Results ===")
   for algo, data in results.items():
       print(f"{algo}: {data['detection']}")

Next Steps
----------

Now that you're familiar with the basics:

- :doc:`tutorial` - Detailed tutorials for each component
- :doc:`user_guide/algorithms` - Learn about specific algorithms
- :doc:`user_guide/visualization` - Visualize watermarking mechanisms
- :doc:`user_guide/evaluation` - Evaluate watermark quality and robustness

