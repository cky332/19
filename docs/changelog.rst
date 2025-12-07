Changelog
=========

All notable changes to MarkDiffusion will be documented in this file.

Version 1.0.0 (2025-01-XX)
--------------------------

Initial Release
~~~~~~~~~~~~~~~

**Implemented Algorithms:**

- Tree-Ring (TR) - Pattern-based watermarking
- Ring-ID (RI) - Multi-key identification
- ROBIN - Robust and invisible watermarking
- WIND - Two-stage robust watermarking  
- SFW - Semantic Fourier watermarking
- Gaussian-Shading (GS) - Performance-lossless watermarking
- GaussMarker (GM) - Dual-domain watermarking
- PRC - Undetectable watermarking
- SEAL - Semantic-aware watermarking
- VideoShield - Video watermarking
- VideoMark - Distortion-free video watermarking

**Features:**

- Unified implementation framework for watermarking algorithms
- Comprehensive evaluation module with 24 tools
- 8 automated evaluation pipelines
- Custom visualization tools for all algorithms
- Support for both image and video watermarking
- Extensive documentation and tutorials

**Evaluation Tools:**

*Detectability:*
- FundamentalSuccessRateCalculator
- DynamicThresholdSuccessRateCalculator

*Image Attacks:*
- JPEG Compression
- Gaussian Blur
- Gaussian Noise
- Rotation
- Crop & Scale
- Brightness Adjustment
- Masking
- Overlay
- Adaptive Noise Injection

*Video Attacks:*
- MPEG-4 Compression
- Frame Averaging
- Frame Swapping
- Video Codec Attack (H.264/H.265/VP9/AV1)
- Frame Rate Adapter
- Frame Interpolation Attack

*Image Quality Metrics:*
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- CLIP Score
- FID (Fréchet Inception Distance)
- Inception Score
- NIQE (Natural Image Quality Evaluator)
- BRISQUE
- VIF (Visual Information Fidelity)
- FSIM (Feature Similarity Index)

*Video Quality Metrics:*
- Subject Consistency
- Background Consistency
- Motion Smoothness
- Dynamic Degree
- Imaging Quality

Recent Updates
--------------

**2025.10.10**

- Added Mask, Overlay, AdaptiveNoiseInjection image attack tools
- Thanks to Zheyu Fu for the contribution

**2025.10.09**

- Added VideoCodecAttack, FrameRateAdapter, FrameInterpolationAttack video attack tools
- Thanks to Luyang Si for the contribution

**2025.10.08**

- Added SSIM, BRISQUE, VIF, FSIM image quality analyzers
- Thanks to Huan Wang for the contribution

**2025.10.07**

- Added SFW (Semantic Fourier Watermarking) algorithm
- Thanks to Huan Wang for the contribution

**2025.10.07**

- Added VideoMark watermarking algorithm
- Thanks to Hanqian Li for the contribution

**2025.09.29**

- Added GaussMarker watermarking algorithm
- Thanks to Luyang Si for the contribution

Upcoming Features
-----------------

**Planned for v1.1.0:**

- Additional watermarking algorithms
- More evaluation metrics
- Enhanced visualization capabilities
- Performance optimizations
- Extended documentation

**Under Consideration:**

- Real-time watermarking support
- Web interface for demonstration
- Pre-trained model zoo
- Integration with more diffusion models
- Support for additional modalities (3D, audio)

Contributing
------------

We welcome contributions! See :doc:`contributing` for guidelines.

To report bugs or request features, please open an issue on GitHub:
https://github.com/THU-BPM/MarkDiffusion/issues

