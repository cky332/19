Citation
========

If you use MarkDiffusion in your research, please cite our paper:

BibTeX
------

.. code-block:: bibtex

   @article{pan2025markdiffusion,
     title={MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models},
     author={Pan, Leyi and Guan, Sheng and Fu, Zheyu and Si, Luyang and Wang, Zian and Hu, Xuming and King, Irwin and Yu, Philip S and Liu, Aiwei and Wen, Lijie},
     journal={arXiv preprint arXiv:2509.10569},
     year={2025}
   }

APA Style
---------

Pan, L., Guan, S., Fu, Z., Si, L., Wang, Z., Hu, X., King, I., Yu, P. S., Liu, A., & Wen, L. (2025). 
MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models. 
*arXiv preprint arXiv:2509.10569*.

MLA Style
---------

Pan, Leyi, et al. "MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models." 
*arXiv preprint arXiv:2509.10569* (2025).

Algorithm-Specific Citations
-----------------------------

If you use specific algorithms, please also cite their original papers:

Tree-Ring Watermark
~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @misc{wen2023treeringwatermarksfingerprintsdiffusion,
      title={Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust}, 
      author={Yuxin Wen and John Kirchenbauer and Jonas Geiping and Tom Goldstein},
      year={2023},
      eprint={2305.20030},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.20030}, 
   }

Ring-ID
~~~~~~~

.. code-block:: bibtex

   @article{ci2024ringid,
      title={RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key Identification},
      author={Ci, Hai and Yang, Pei and Song, Yiren and Shou, Mike Zheng},
      journal={arXiv preprint arXiv:2404.14055},
      year={2024}
   }

ROBIN
~~~~~

.. code-block:: bibtex

   @inproceedings{huangrobin,
      title={ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization},
      author={Huang, Huayang and Wu, Yu and Wang, Qian},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
   }


WIND
~~~~

.. code-block:: bibtex

   @article{arabi2024hidden,
      title={Hidden in the Noise: Two-Stage Robust Watermarking for Images},
      author={Arabi, Kasra and Feuer, Benjamin and Witter, R Teal and Hegde, Chinmay and Cohen, Niv},
      journal={arXiv preprint arXiv:2412.04653},
      year={2024}
   }

SFW
~~~

.. code-block:: bibtex

   @inproceedings{lee2025semantic,
      title={Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity},
      author={Lee, Sung Ju and Cho, Nam Ik},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages={18759--18769},
      year={2025}
   }

Gaussian-Shading
~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @article{yang2024gaussian,
      title={Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models}, 
      author={Yang, Zijin and Zeng, Kai and Chen, Kejiang and Fang, Han and Zhang, Weiming and Yu, Nenghai},
      journal={arXiv preprint arXiv:2404.04956},
      year={2024},
   }

GaussMarker
~~~~~~~~~~~

.. code-block:: bibtex

   @misc{li2025gaussmarkerrobustdualdomainwatermark,
      title={GaussMarker: Robust Dual-Domain Watermark for Diffusion Models}, 
      author={Kecen Li and Zhicong Huang and Xinwen Hou and Cheng Hong},
      year={2025},
      eprint={2506.11444},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2506.11444}, 
   }

PRC
~~~

.. code-block:: bibtex

   @article{gunn2025undetectable,
      title={An undetectable watermark for generative image models},
      author={Gunn, Sam and Zhao, Xuandong and Song, Dawn},
      journal={arXiv preprint arXiv:2410.07369},
      year={2024}
   }

SEAL
~~~~

.. code-block:: bibtex

   @article{arabi2025seal,
      title={SEAL: Semantic Aware Image Watermarking},
      author={Arabi, Kasra and Witter, R Teal and Hegde, Chinmay and Cohen, Niv},
      journal={arXiv preprint arXiv:2503.12172},
      year={2025}
   }

VideoShield
~~~~~~~~~~~

.. code-block:: bibtex

   @inproceedings{hu2025videoshield,
      title={VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking}, 
      author={Runyi Hu and Jie Zhang and Yiming Li and Jiwei Li and Qing Guo and Han Qiu and Tianwei Zhang},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2025}
   }

VideoMark
~~~~~~~~~

.. code-block:: bibtex

   @article{hu2025videomark,
      title={VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models},
      author={Hu, Xuming and Li, Hanqian and Li, Jungang and Liu, Aiwei},
      journal={arXiv preprint arXiv:2504.16359},
      year={2025}
   }

Acknowledgments
---------------

We would like to thank:

- All contributors to the MarkDiffusion project
- The authors of the watermarking algorithms implemented in this toolkit
- The open-source community for their valuable feedback and contributions
- Research institutions supporting this work

Using MarkDiffusion in Publications
------------------------------------

When using MarkDiffusion in your research:

1. **Cite the main MarkDiffusion paper** (required)
2. **Cite specific algorithm papers** you use (required)
3. **Mention the toolkit in your acknowledgments**
4. **Link to the GitHub repository**

Example acknowledgment text:

   *"This research utilized MarkDiffusion [1], an open-source toolkit for generative 
   watermarking. We specifically employed the Gaussian-Shading algorithm [2] for 
   watermark embedding and detection."*

License
-------

MarkDiffusion is released under the MIT License. See the LICENSE file for details.

When using MarkDiffusion, please ensure compliance with the licenses of:

- Individual watermarking algorithms
- Pre-trained models
- Datasets used for evaluation

Contact
-------

For questions about citation or collaboration:

- **GitHub**: https://github.com/THU-BPM/MarkDiffusion
- **Paper**: https://arxiv.org/abs/2509.10569
- **Homepage**: https://generative-watermark.github.io/

Updates
-------

This citation information was last updated: November 2025

For the most up-to-date citation information, please check:

- The project README
- The paper on arXiv
- The project homepage

