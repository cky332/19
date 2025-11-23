"""
Pytest configuration and fixtures for MarkDiffusion watermark algorithm tests.

This file contains all pytest hooks, fixtures, and configuration that will be
automatically discovered and used by pytest.
"""

import pytest
import torch
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image

from watermark.auto_watermark import AutoWatermark, PIPELINE_SUPPORTED_WATERMARKS
from utils.diffusion_config import DiffusionConfig
from diffusers import (
    StableDiffusionPipeline,
    TextToVideoSDPipeline,
    DPMSolverMultistepScheduler
)
from utils.pipeline_utils import (
    PIPELINE_TYPE_IMAGE,
    PIPELINE_TYPE_TEXT_TO_VIDEO,
    PIPELINE_TYPE_IMAGE_TO_VIDEO
)


# ============================================================================
# Test Configuration
# ============================================================================

# Default model paths (can be overridden via pytest options)
DEFAULT_IMAGE_MODEL_PATH = "stabilityai/stable-diffusion-2-1-base"
DEFAULT_VIDEO_MODEL_PATH = "damo-vilab/text-to-video-ms-1.7b"

# Test prompts
TEST_PROMPT_IMAGE = "A beautiful sunset over the ocean"
TEST_PROMPT_VIDEO = "A cinematic timelapse of city lights at night"

# Test parameters
IMAGE_SIZE = (512, 512)
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
GEN_SEED = 42
NUM_FRAMES = 16


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--algorithm",
        action="store",
        default=None,
        help="Specific algorithm to test (e.g., TR, GS, VideoShield)"
    )
    parser.addoption(
        "--image-model-path",
        action="store",
        default=DEFAULT_IMAGE_MODEL_PATH,
        help="Path to image generation model"
    )
    parser.addoption(
        "--video-model-path",
        action="store",
        default=DEFAULT_VIDEO_MODEL_PATH,
        help="Path to video generation model"
    )
    parser.addoption(
        "--skip-generation",
        action="store_true",
        default=False,
        help="Skip generation tests (only test detection)"
    )
    parser.addoption(
        "--skip-detection",
        action="store_true",
        default=False,
        help="Skip detection tests (only test generation)"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "image: mark test as image watermark test")
    config.addinivalue_line("markers", "video: mark test as video watermark test")
    config.addinivalue_line("markers", "inversion: mark test as inversion module test")
    config.addinivalue_line("markers", "visualization: mark test as visualization test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "dataset: mark test as dataset test")
    config.addinivalue_line("markers", "pipeline: mark test as pipeline test")
    config.addinivalue_line("markers", "detection: mark test as detection pipeline test")
    config.addinivalue_line("markers", "quality: mark test as quality analysis pipeline test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "summary: mark test as summary test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    algorithm = config.getoption("--algorithm")
    if algorithm:
        # Filter tests to only run for specified algorithm
        selected = []
        for item in items:
            if algorithm in item.nodeid:
                selected.append(item)
        items[:] = selected


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information to pytest output."""
    terminalreporter.write_sep("=", "Watermark Algorithm Test Summary")

    # Count passed/failed tests by algorithm
    passed = terminalreporter.stats.get('passed', [])
    failed = terminalreporter.stats.get('failed', [])
    skipped = terminalreporter.stats.get('skipped', [])

    terminalreporter.write_line(f"Passed: {len(passed)}")
    terminalreporter.write_line(f"Failed: {len(failed)}")
    terminalreporter.write_line(f"Skipped: {len(skipped)}")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get the device for testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope="session")
def image_model_path(request):
    """Get the image model path from command line or use default."""
    return request.config.getoption("--image-model-path")


@pytest.fixture(scope="session")
def video_model_path(request):
    """Get the video model path from command line or use default."""
    return request.config.getoption("--video-model-path")


@pytest.fixture(scope="session")
def skip_generation(request):
    """Check if generation tests should be skipped."""
    return request.config.getoption("--skip-generation")


@pytest.fixture(scope="session")
def skip_detection(request):
    """Check if detection tests should be skipped."""
    return request.config.getoption("--skip-detection")


@pytest.fixture(scope="session")
def image_pipeline(device, image_model_path):
    """Create and cache image generation pipeline."""
    try:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            image_model_path,
            subfolder="scheduler"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            image_model_path,
            scheduler=scheduler
        ).to(device)
        return pipe, scheduler
    except Exception as e:
        pytest.skip(f"Failed to load image model: {e}")


@pytest.fixture(scope="session")
def video_pipeline(device, video_model_path):
    """Create and cache video generation pipeline."""
    try:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            video_model_path,
            subfolder="scheduler"
        )
        pipe = TextToVideoSDPipeline.from_pretrained(
            video_model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)
        return pipe, scheduler
    except Exception as e:
        pytest.skip(f"Failed to load video model: {e}")


@pytest.fixture
def image_diffusion_config(device, image_pipeline):
    """Create diffusion config for image generation."""
    pipe, scheduler = image_pipeline
    return DiffusionConfig(
        scheduler=scheduler,
        pipe=pipe,
        device=device,
        image_size=IMAGE_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        gen_seed=GEN_SEED,
        inversion_type="ddim"
    )


@pytest.fixture
def video_diffusion_config(device, video_pipeline):
    """Create diffusion config for video generation."""
    pipe, scheduler = video_pipeline
    return DiffusionConfig(
        scheduler=scheduler,
        pipe=pipe,
        device=device,
        image_size=IMAGE_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        gen_seed=GEN_SEED,
        inversion_type="ddim",
        num_frames=NUM_FRAMES
    )


# Export constants for use in test files
__all__ = [
    'TEST_PROMPT_IMAGE',
    'TEST_PROMPT_VIDEO',
    'IMAGE_SIZE',
    'NUM_INFERENCE_STEPS',
    'GUIDANCE_SCALE',
    'GEN_SEED',
    'NUM_FRAMES',
]
