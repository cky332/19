Quick Start
===========

Google Colab Demo
-----------------

If you're interested in trying out MarkDiffusion without installing anything, you can use 
`Google Colab <https://colab.research.google.com/drive/1N1C9elDAB5zwF4FxKKYMCqR3eSpCSqAW?usp=sharing>`_ 
to see how it works.

Installation
------------

**(Recommended)** We released PyPI package for MarkDiffusion. You can install it directly with pip:

.. code-block:: bash

   conda create -n markdiffusion python=3.11
   conda activate markdiffusion
   pip install markdiffusion[optional]

**(Alternative)** For users who are restricted only to use conda environment, we also provide a 
conda-forge package, which can be installed with the following commands:

.. code-block:: bash

   conda create -n markdiffusion python=3.11
   conda activate markdiffusion
   conda config --add channels conda-forge
   conda config --set channel_priority strict
   conda install markdiffusion

.. note::
   Some advanced features require additional packages that are not available on conda and cannot 
   be included in the release. You will need to install those separately if necessary.

How to Use the Toolkit
----------------------

After installation, there are two ways to use MarkDiffusion:

Method 1: Clone Repository for Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Clone the repository to try the demos or use it for custom development.** 

The ``MarkDiffusion_demo.ipynb`` notebook offers detailed demonstrations for various use cases—please 
review it for guidance. Here's a quick example of generating and detecting watermarked image with the 
TR algorithm:

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
   watermarked_image.save("watermarked_image.png")

   # Detect watermark
   detection_result = watermark.detect_watermark_in_media(watermarked_image)
   print(f"Watermark detected: {detection_result}")

Method 2: Import as Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Import markdiffusion library directly in your code without cloning the repository.** 

The ``MarkDiffusion_pypi_demo.ipynb`` notebook provides comprehensive examples for using MarkDiffusion 
via the markdiffusion library—please review it for guidance. Here's a quick example:

.. code-block:: python

   import torch
   from markdiffusion.watermark import AutoWatermark
   from markdiffusion.utils import DiffusionConfig
   from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

   # Device
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Using device: {device}")

   # Model path
   MODEL_PATH = "huanzi05/stable-diffusion-2-1-base"

   # Initialize scheduler and pipeline
   scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
   pipe = StableDiffusionPipeline.from_pretrained(
       MODEL_PATH,
       scheduler=scheduler,
       torch_dtype=torch.float16 if device == "cuda" else torch.float32,
       safety_checker=None,
   ).to(device)

   # Create DiffusionConfig for image generation
   image_diffusion_config = DiffusionConfig(
       scheduler=scheduler,
       pipe=pipe,
       device=device,
       image_size=(512, 512),
       guidance_scale=7.5,
       num_inference_steps=50,
       gen_seed=42,
       inversion_type="ddim"
   )

   # Load Tree-Ring watermark algorithm
   tr_watermark = AutoWatermark.load('TR', diffusion_config=image_diffusion_config)
   print("TR watermark algorithm loaded successfully!")

   # Generate watermarked image
   prompt = "A beautiful landscape with mountains and a river at sunset"

   watermarked_image = tr_watermark.generate_watermarked_media(input_data=prompt)

   # Display the watermarked image
   watermarked_image.save("watermarked_image.png")
   print("Watermarked image generated!")

   # Detect watermark in the watermarked image
   detection_result = tr_watermark.detect_watermark_in_media(watermarked_image)
   print("Watermarked image detection result:")
   print(detection_result)

Next Steps
----------

Now that you're familiar with the basics, explore more:

- :doc:`user_guide/algorithms` - Learn about specific algorithms
- :doc:`user_guide/visualization` - Visualize watermarking mechanisms
- :doc:`user_guide/evaluation` - Evaluate watermark quality and robustness
