# Copyright 2025 THU-BPM MarkDiffusion.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
from visualize.base import BaseVisualizer
from visualize.data_for_visualization import DataForVisualization


class VideoMarkVisualizer(BaseVisualizer):
    """VideoMark watermark visualization class.
    
    This visualizer handles watermark visualization for VideoShield algorithm,
    which extends Gaussian Shading to the video domain by adding frame dimensions.
    
    Key Members for VideoMarkVisualizer:
        - self.data.orig_watermarked_latents: [B, C, F, H, W]
        - self.data.reversed_latents: List[[B, C, F, H, W]]
    """
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1, is_video: bool = True):
        super().__init__(data_for_visualization, dpi, watermarking_step, is_video)
    
    def draw_watermarked_video_frames(self,
                                    num_frames: int = 4,
                                    title: str = "Watermarked Video Frames",
                                    ax: Axes | None = None) -> Axes:
        """
        Draw multiple frames from the watermarked video.

        DEPRECATED:
            This method is deprecated and will be removed in a future version.
            Please use `draw_watermarked_image` instead.

        This method displays a grid of video frames to show the temporal
        consistency of the watermarked video.

        Args:
            num_frames: Number of frames to display (default: 4)
            title: The title of the plot
            ax: The axes to plot on

        Returns:
            The plotted axes
        """
        return self._draw_video_frames(
            title=title,
            num_frames=num_frames,
            ax=ax
        )
