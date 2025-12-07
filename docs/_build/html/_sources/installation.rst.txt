Installation
============

This guide will help you install MarkDiffusion and its dependencies.

Requirements
------------

- Python 3.10 or higher
- PyTorch (with CUDA support recommended for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for video watermarking)
- CUDA-compatible GPU (optional but highly recommended)

Basic Installation
------------------

1. **Clone the Repository**

   .. code-block:: bash

      git clone https://github.com/THU-BPM/MarkDiffusion.git
      cd MarkDiffusion

2. **Install Dependencies**

   .. code-block:: bash

      pip install -r requirements.txt

3. **Download Pre-trained Models**

   MarkDiffusion uses pre-trained models stored on Hugging Face. Download the required models:

   .. code-block:: bash

      # The models will be downloaded to the ckpts/ directory
      # Visit: https://huggingface.co/Generative-Watermark-Toolkits

   For each algorithm you plan to use, download the corresponding model weights from the 
   `Generative-Watermark-Toolkits <https://huggingface.co/Generative-Watermark-Toolkits>`_ 
   repository and place them in the appropriate ``ckpts/`` subdirectory.

Installation with Conda
-----------------------

If you prefer using Conda for environment management:

.. code-block:: bash

   # Create a new conda environment
   conda create -n markdiffusion python=3.10
   conda activate markdiffusion

   # Install PyTorch with CUDA support
   conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia

   # Install other dependencies
   pip install -r requirements.txt

GPU Support
-----------

For GPU acceleration, make sure you have:

1. **NVIDIA GPU** with CUDA support
2. **CUDA Toolkit** installed (version 11.8 or higher)
3. **cuDNN** library

To verify GPU availability:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"Device count: {torch.cuda.device_count()}")

Algorithm-Specific Setup
------------------------

Some algorithms require additional model checkpoints beyond the base installation:

GaussMarker (GM)
~~~~~~~~~~~~~~~~

GaussMarker requires two pre-trained models for watermark detection and restoration:

1. **GNR Model** (Generative Noise Restoration): A UNet-based model for restoring watermark bits from noisy latents
2. **Fuser Model**: A classifier for fusion-based watermark detection decisions

**Setup:**

.. code-block:: bash

   # Create the checkpoint directory
   mkdir -p watermark/gm/ckpts/
   
   # Download models from Hugging Face
   # Visit: https://huggingface.co/Generative-Watermark-Toolkits/GaussMarker
   # Place the following files:
   #   - model_final.pth -> watermark/gm/ckpts/model_final.pth
   #   - sd21_cls2.pkl -> watermark/gm/ckpts/sd21_cls2.pkl

**Configuration Path** (in ``config/GM.json``):

- ``gnr_checkpoint``: ``"watermark/gm/ckpts/model_final.pth"``
- ``fuser_checkpoint``: ``"watermark/gm/ckpts/sd21_cls2.pkl"``

SEAL
~~~~

SEAL uses pre-trained models from Hugging Face for caption generation and embedding:

1. **BLIP2 Model**: For generating image captions (blip2-flan-t5-xl)
2. **Sentence Transformer**: For caption embedding

**Setup:**

.. code-block:: bash

   # These models will be automatically downloaded from Hugging Face on first use
   # Or you can pre-download them:
   
   # Download BLIP2 model
   python -c "from transformers import Blip2Processor, Blip2ForConditionalGeneration; \
   Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl'); \
   Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl')"
   
   # Download sentence transformer (if using custom fine-tuned model)
   # Update config/SEAL.json paths accordingly

**Configuration** (in ``config/SEAL.json``):

- ``cap_processor``: Path or model name for BLIP2 processor
- ``cap_model``: Path or model name for BLIP2 model
- ``sentence_model``: Path or model name for sentence transformer

.. note::
   SEAL models are large (~15GB for BLIP2). Ensure you have sufficient disk space and memory.

Other Algorithms
~~~~~~~~~~~~~~~~

The following algorithms work with the base installation and do not require additional checkpoints:

- **Tree-Ring (TR)**, **Ring-ID (RI)**, **ROBIN**, **WIND**, **SFW**: Pattern-based methods using frequency domain manipulation
- **Gaussian-Shading (GS)**, **PRC**: Key-based methods with built-in watermark generation
- **VideoShield**, **VideoMark**: Video watermarking algorithms using temporal consistency

Verification
------------

To verify your installation, run the test suite:

.. code-block:: bash

   python -m pytest test/

Or try a simple example:

.. code-block:: python

   from watermark.auto_watermark import AutoWatermark
   print("MarkDiffusion successfully installed!")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue: Module not found error**

Solution: Make sure all dependencies are installed:

.. code-block:: bash

   pip install -r requirements.txt --upgrade

**Issue: Model weights not found**

Solution: Download the required models from Hugging Face and place them in the correct directory structure.

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/THU-BPM/MarkDiffusion/issues>`_
2. Consult the `FAQ <https://github.com/THU-BPM/MarkDiffusion/wiki/FAQ>`_
3. Open a new issue with detailed information about your problem

Next Steps
----------

Now that you have installed MarkDiffusion, proceed to:

- :doc:`quickstart` - Get started with basic examples
- :doc:`tutorial` - Learn through step-by-step tutorials
- :doc:`user_guide/algorithms` - Explore available algorithms

