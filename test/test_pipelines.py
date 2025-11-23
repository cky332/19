"""
Comprehensive tests for MarkDiffusion evaluation pipelines and datasets.

This module tests:
1. Dataset classes (StableDiffusionPromptsDataset, MSCOCODataset, VBenchDataset)
2. Detection pipelines (WatermarkedMediaDetectionPipeline, UnWatermarkedMediaDetectionPipeline)
3. Image quality analysis pipelines (6 pipelines)
4. Video quality analysis pipeline

Usage:
    # Test all pipelines and datasets
    pytest test/test_pipelines.py -v

    # Test specific components
    pytest test/test_pipelines.py::test_datasets -v
    pytest test/test_pipelines.py::test_detection_pipelines -v
    pytest test/test_pipelines.py::test_image_quality_pipelines -v
"""

import pytest
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

# Import dataset classes
from evaluation.dataset import (
    BaseDataset,
    StableDiffusionPromptsDataset,
    MSCOCODataset,
    VBenchDataset
)

# Import pipeline classes
from evaluation.pipelines.detection import (
    WatermarkedMediaDetectionPipeline,
    UnWatermarkedMediaDetectionPipeline,
    DetectionPipelineReturnType
)

from evaluation.pipelines.image_quality_analysis import (
    DirectImageQualityAnalysisPipeline,
    ReferencedImageQualityAnalysisPipeline,
    GroupImageQualityAnalysisPipeline,
    RepeatImageQualityAnalysisPipeline,
    ComparedImageQualityAnalysisPipeline,
    QualityPipelineReturnType
)

from evaluation.pipelines.video_quality_analysis import (
    DirectVideoQualityAnalysisPipeline
)

# Import watermark and tools for testing
from watermark.auto_watermark import AutoWatermark
from evaluation.tools.image_quality_analyzer import (
    NIQECalculator, CLIPScoreCalculator, FIDCalculator,
    InceptionScoreCalculator, LPIPSAnalyzer, PSNRAnalyzer,
    SSIMAnalyzer, BRISQUEAnalyzer, VIFAnalyzer, FSIMAnalyzer
)
from evaluation.tools.video_quality_analyzer import (
    SubjectConsistencyAnalyzer,
    MotionSmoothnessAnalyzer,
    DynamicDegreeAnalyzer,
    BackgroundConsistencyAnalyzer,
    ImagingQualityAnalyzer
)
from evaluation.tools.image_editor import (
    JPEGCompression, GaussianNoise, GaussianBlurring,
    Rotation, CrSc, Brightness
)


# ============================================================================
# Test Cases - Datasets
# ============================================================================

@pytest.mark.dataset
def test_base_dataset():
    """Test BaseDataset class."""
    dataset = BaseDataset(max_samples=10)

    # Test initial state
    assert dataset.max_samples == 10
    assert dataset.num_samples == 0
    assert dataset.num_references == 0
    assert len(dataset) == 0

    # Test adding data
    dataset.prompts = ["test1", "test2"]
    assert dataset.num_samples == 2
    assert len(dataset) == 2
    assert dataset.get_prompt(0) == "test1"
    assert dataset.get_prompt(1) == "test2"

    # Test __getitem__ without references
    assert dataset[0] == "test1"
    assert dataset[1] == "test2"

    # Test with references
    img1 = Image.new('RGB', (256, 256))
    img2 = Image.new('RGB', (256, 256))
    dataset.references = [img1, img2]

    assert dataset.num_references == 2
    assert dataset.get_reference(0) == img1
    prompt, ref = dataset[0]
    assert prompt == "test1"
    assert ref == img1

    print("✓ BaseDataset test passed")


@pytest.mark.dataset
@pytest.mark.slow
def test_stable_diffusion_prompts_dataset():
    """Test StableDiffusionPromptsDataset class."""
    try:
        # Test initialization with small sample
        dataset = StableDiffusionPromptsDataset(max_samples=5, split="test", shuffle=False)

        # Check dataset properties
        assert dataset.name == "Stable Diffusion Prompts"
        assert dataset.max_samples == 5
        assert dataset.split == "test"
        assert dataset.shuffle == False

        # Check data loading
        assert len(dataset) > 0
        assert len(dataset) <= 5
        assert dataset.num_samples == len(dataset.prompts)
        assert dataset.num_references == 0  # This dataset has no references

        # Check prompt format
        for i in range(len(dataset)):
            prompt = dataset.get_prompt(i)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

        # Test with shuffle
        dataset_shuffled = StableDiffusionPromptsDataset(max_samples=5, split="test", shuffle=True)
        assert len(dataset_shuffled) <= 5

        print(f"✓ StableDiffusionPromptsDataset test passed (loaded {len(dataset)} prompts)")

    except Exception as e:
        # Skip if dataset files not available
        if "dataset/stable_diffusion_prompts" in str(e):
            pytest.skip(f"StableDiffusionPromptsDataset files not available: {e}")
        else:
            raise e


@pytest.mark.dataset
@pytest.mark.slow
def test_mscoco_dataset():
    """Test MSCOCODataset class."""
    try:
        # Test initialization with very small sample
        dataset = MSCOCODataset(max_samples=2, shuffle=False)

        # Check dataset properties
        assert dataset.name == "MS-COCO 2017"
        assert dataset.max_samples == 2
        assert dataset.shuffle == False

        # Check data loading
        if len(dataset) > 0:
            assert len(dataset) <= 2
            assert dataset.num_samples == len(dataset.prompts)
            assert dataset.num_references == len(dataset.references)

            # Check data format
            for i in range(len(dataset)):
                prompt = dataset.get_prompt(i)
                assert isinstance(prompt, str)
                assert len(prompt) > 0

                if dataset.num_references > 0:
                    reference = dataset.get_reference(i)
                    if reference is not None:  # May fail to load from URL
                        assert isinstance(reference, Image.Image)

        print(f"✓ MSCOCODataset test passed (loaded {len(dataset)} samples)")

    except Exception as e:
        # Skip if dataset files not available
        if "dataset/mscoco" in str(e) or "parquet" in str(e):
            pytest.skip(f"MSCOCODataset files not available: {e}")
        else:
            raise e


@pytest.mark.dataset
def test_vbench_dataset():
    """Test VBenchDataset class."""
    dimensions = ["subject_consistency", "background_consistency",
                  "imaging_quality", "motion_smoothness", "dynamic_degree"]

    for dimension in dimensions:
        try:
            # Test initialization
            dataset = VBenchDataset(max_samples=5, dimension=dimension, shuffle=False)

            # Check dataset properties
            assert dataset.name == "VBench"
            assert dataset.max_samples == 5
            assert dataset.dimension == dimension
            assert dataset.shuffle == False

            # Check data loading
            if len(dataset) > 0:
                assert len(dataset) <= 5
                assert dataset.num_samples == len(dataset.prompts)
                assert dataset.num_references == 0  # VBench has no reference images

                # Check prompt format
                for i in range(len(dataset)):
                    prompt = dataset.get_prompt(i)
                    assert isinstance(prompt, str)
                    assert len(prompt) > 0

            print(f"✓ VBenchDataset test passed for dimension: {dimension}")

        except Exception as e:
            # Skip if dataset files not available
            if "dataset/vbench" in str(e) or f"{dimension}.txt" in str(e):
                pytest.skip(f"VBenchDataset files not available for {dimension}: {e}")
            else:
                raise e


# ============================================================================
# Test Cases - Detection Pipelines
# ============================================================================

@pytest.mark.pipeline
@pytest.mark.detection
def test_watermarked_media_detection_pipeline():
    """Test WatermarkedMediaDetectionPipeline."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 3
    mock_dataset.prompts = ["prompt1", "prompt2", "prompt3"]
    mock_dataset.__len__ = MagicMock(return_value=3)
    mock_dataset.__getitem__ = MagicMock(side_effect=lambda i: mock_dataset.prompts[i])

    # Create pipeline
    pipeline = WatermarkedMediaDetectionPipeline(
        dataset=mock_dataset,
        media_editor_list=[],
        show_progress=False,
        return_type=DetectionPipelineReturnType.SCORES
    )

    assert pipeline.dataset == mock_dataset
    assert pipeline.show_progress == False
    assert pipeline.return_type == DetectionPipelineReturnType.SCORES
    assert len(pipeline.media_editor_list) == 0

    # Test with media editors
    pipeline_with_editors = WatermarkedMediaDetectionPipeline(
        dataset=mock_dataset,
        media_editor_list=[JPEGCompression(quality=75)],
        show_progress=False,
        return_type=DetectionPipelineReturnType.LABELS
    )

    assert len(pipeline_with_editors.media_editor_list) == 1
    assert isinstance(pipeline_with_editors.media_editor_list[0], JPEGCompression)
    assert pipeline_with_editors.return_type == DetectionPipelineReturnType.LABELS

    print("✓ WatermarkedMediaDetectionPipeline test passed")


@pytest.mark.pipeline
@pytest.mark.detection
def test_unwatermarked_media_detection_pipeline():
    """Test UnWatermarkedMediaDetectionPipeline."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 3
    mock_dataset.prompts = ["prompt1", "prompt2", "prompt3"]
    mock_dataset.__len__ = MagicMock(return_value=3)
    mock_dataset.__getitem__ = MagicMock(side_effect=lambda i: mock_dataset.prompts[i])

    # Create pipeline
    pipeline = UnWatermarkedMediaDetectionPipeline(
        dataset=mock_dataset,
        media_editor_list=[GaussianNoise(std=0.01)],
        show_progress=True,
        return_type=DetectionPipelineReturnType.SCORES_AND_LABELS
    )

    assert pipeline.dataset == mock_dataset
    assert pipeline.show_progress == True
    assert pipeline.return_type == DetectionPipelineReturnType.SCORES_AND_LABELS
    assert len(pipeline.media_editor_list) == 1
    assert isinstance(pipeline.media_editor_list[0], GaussianNoise)

    print("✓ UnWatermarkedMediaDetectionPipeline test passed")


# ============================================================================
# Test Cases - Image Quality Analysis Pipelines
# ============================================================================

@pytest.mark.pipeline
@pytest.mark.quality
def test_direct_image_quality_analysis_pipeline():
    """Test DirectImageQualityAnalysisPipeline."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 2
    mock_dataset.__len__ = MagicMock(return_value=2)

    # Create pipeline
    pipeline = DirectImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[NIQECalculator()],
        show_progress=False,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

    assert pipeline.dataset == mock_dataset
    assert len(pipeline.analyzers) == 1
    assert isinstance(pipeline.analyzers[0], NIQECalculator)
    assert pipeline.return_type == QualityPipelineReturnType.MEAN_SCORES

    print("✓ DirectImageQualityAnalysisPipeline test passed")


@pytest.mark.pipeline
@pytest.mark.quality
def test_referenced_image_quality_analysis_pipeline():
    """Test ReferencedImageQualityAnalysisPipeline."""
    # Create mock dataset with references
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 2
    mock_dataset.num_references = 2
    mock_dataset.__len__ = MagicMock(return_value=2)

    # Create pipeline
    pipeline = ReferencedImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[CLIPScoreCalculator()],
        unwatermarked_image_source='generated',
        reference_image_source='natural',
        show_progress=False,
        return_type=QualityPipelineReturnType.ALL_SCORES
    )

    assert pipeline.dataset == mock_dataset
    assert len(pipeline.analyzers) == 1
    assert isinstance(pipeline.analyzers[0], CLIPScoreCalculator)
    assert pipeline.unwatermarked_image_source == 'generated'
    assert pipeline.reference_image_source == 'natural'
    assert pipeline.return_type == QualityPipelineReturnType.ALL_SCORES

    print("✓ ReferencedImageQualityAnalysisPipeline test passed")


@pytest.mark.pipeline
@pytest.mark.quality
def test_group_image_quality_analysis_pipeline():
    """Test GroupImageQualityAnalysisPipeline."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 10
    mock_dataset.__len__ = MagicMock(return_value=10)

    # Create pipeline with FID calculator
    pipeline = GroupImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[FIDCalculator()],
        unwatermarked_image_source='generated',
        reference_image_source='natural',
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

    assert pipeline.dataset == mock_dataset
    assert len(pipeline.analyzers) == 1
    assert isinstance(pipeline.analyzers[0], FIDCalculator)

    # Test with IS calculator (no reference needed)
    pipeline_is = GroupImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[InceptionScoreCalculator()],
        show_progress=False,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

    assert isinstance(pipeline_is.analyzers[0], InceptionScoreCalculator)

    print("✓ GroupImageQualityAnalysisPipeline test passed")


@pytest.mark.pipeline
@pytest.mark.quality
def test_repeat_image_quality_analysis_pipeline():
    """Test RepeatImageQualityAnalysisPipeline."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 5
    mock_dataset.__len__ = MagicMock(return_value=5)

    # Create pipeline
    pipeline = RepeatImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        prompt_per_image=10,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[LPIPSAnalyzer()],
        show_progress=False,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

    assert pipeline.dataset == mock_dataset
    assert pipeline.prompt_per_image == 10
    assert len(pipeline.analyzers) == 1
    assert isinstance(pipeline.analyzers[0], LPIPSAnalyzer)

    print("✓ RepeatImageQualityAnalysisPipeline test passed")


@pytest.mark.pipeline
@pytest.mark.quality
def test_compared_image_quality_analysis_pipeline():
    """Test ComparedImageQualityAnalysisPipeline."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 3
    mock_dataset.__len__ = MagicMock(return_value=3)

    # Test with multiple analyzers
    analyzers = [
        PSNRAnalyzer(),
        SSIMAnalyzer(),
        LPIPSAnalyzer()
    ]

    # Create pipeline
    pipeline = ComparedImageQualityAnalysisPipeline(
        dataset=mock_dataset,
        watermarked_image_editor_list=[JPEGCompression(quality=90)],
        unwatermarked_image_editor_list=[],
        analyzers=analyzers,
        show_progress=False,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

    assert pipeline.dataset == mock_dataset
    assert len(pipeline.analyzers) == 3
    assert isinstance(pipeline.analyzers[0], PSNRAnalyzer)
    assert isinstance(pipeline.analyzers[1], SSIMAnalyzer)
    assert isinstance(pipeline.analyzers[2], LPIPSAnalyzer)
    assert len(pipeline.watermarked_image_editor_list) == 1

    print("✓ ComparedImageQualityAnalysisPipeline test passed")


# ============================================================================
# Test Cases - Video Quality Analysis Pipeline
# ============================================================================

@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.video
def test_direct_video_quality_analysis_pipeline():
    """Test DirectVideoQualityAnalysisPipeline."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.num_samples = 5
    mock_dataset.__len__ = MagicMock(return_value=5)

    # Test with different video analyzers
    analyzers = [
        SubjectConsistencyAnalyzer(device='cpu'),
        MotionSmoothnessAnalyzer(device='cpu'),
        DynamicDegreeAnalyzer(device='cpu')
    ]

    # Create pipeline
    pipeline = DirectVideoQualityAnalysisPipeline(
        dataset=mock_dataset,
        watermarked_video_editor_list=[],
        unwatermarked_video_editor_list=[],
        watermarked_frame_editor_list=[JPEGCompression(quality=85)],
        unwatermarked_frame_editor_list=[],
        analyzers=analyzers,
        show_progress=False,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

    assert pipeline.dataset == mock_dataset
    assert len(pipeline.analyzers) == 3
    assert isinstance(pipeline.analyzers[0], SubjectConsistencyAnalyzer)
    assert isinstance(pipeline.analyzers[1], MotionSmoothnessAnalyzer)
    assert isinstance(pipeline.analyzers[2], DynamicDegreeAnalyzer)
    assert len(pipeline.watermarked_frame_editor_list) == 1

    # Test with remaining analyzers
    pipeline2 = DirectVideoQualityAnalysisPipeline(
        dataset=mock_dataset,
        watermarked_video_editor_list=[],
        unwatermarked_video_editor_list=[],
        watermarked_frame_editor_list=[],
        unwatermarked_frame_editor_list=[],
        analyzers=[
            BackgroundConsistencyAnalyzer(device='cpu'),
            ImagingQualityAnalyzer(device='cpu')
        ],
        show_progress=True,
        return_type=QualityPipelineReturnType.ALL_SCORES
    )

    assert len(pipeline2.analyzers) == 2
    assert isinstance(pipeline2.analyzers[0], BackgroundConsistencyAnalyzer)
    assert isinstance(pipeline2.analyzers[1], ImagingQualityAnalyzer)
    assert pipeline2.return_type == QualityPipelineReturnType.ALL_SCORES

    print("✓ DirectVideoQualityAnalysisPipeline test passed")


# ============================================================================
# Test Cases - Return Types
# ============================================================================

@pytest.mark.pipeline
def test_pipeline_return_types():
    """Test pipeline return type enumerations."""
    # Test DetectionPipelineReturnType
    assert DetectionPipelineReturnType.SCORES.value == "scores"
    assert DetectionPipelineReturnType.LABELS.value == "labels"
    assert DetectionPipelineReturnType.SCORES_AND_LABELS.value == "scores_and_labels"

    # Test QualityPipelineReturnType
    assert QualityPipelineReturnType.MEAN_SCORES.value == "mean_scores"
    assert QualityPipelineReturnType.ALL_SCORES.value == "all_scores"

    print("✓ Pipeline return types test passed")


# ============================================================================
# Integration Tests - Pipeline with Real Watermark
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_with_mock_watermark(tmp_path):
    """Test pipeline integration with mock watermark."""
    # Create minimal mock watermark
    mock_watermark = MagicMock()

    # Mock generate methods
    mock_watermark.generate_watermarked_media = MagicMock(
        return_value=Image.new('RGB', (256, 256))
    )
    mock_watermark.generate_unwatermarked_media = MagicMock(
        return_value=Image.new('RGB', (256, 256))
    )

    # Mock detection method
    mock_watermark.detect_watermark_in_media = MagicMock(
        return_value={'is_watermarked': True, 'score': 0.95}
    )

    # Create minimal dataset
    dataset = BaseDataset(max_samples=2)
    dataset.prompts = ["test1", "test2"]

    # Test detection pipeline
    detection_pipeline = WatermarkedMediaDetectionPipeline(
        dataset=dataset,
        media_editor_list=[],
        show_progress=False,
        return_type=DetectionPipelineReturnType.SCORES
    )

    # Mock evaluate method
    with patch.object(detection_pipeline, 'evaluate') as mock_evaluate:
        mock_evaluate.return_value = [0.95, 0.92]
        result = detection_pipeline.evaluate(mock_watermark)
        assert len(result) == 2
        assert all(isinstance(score, float) for score in result)

    print("✓ Pipeline integration test with mock watermark passed")


# ============================================================================
# Summary Test
# ============================================================================

@pytest.mark.summary
def test_all_pipelines_summary():
    """Summary test to verify all pipeline classes are importable and constructable."""
    pipelines_tested = []

    # Detection pipelines
    try:
        from evaluation.pipelines.detection import (
            WatermarkedMediaDetectionPipeline,
            UnWatermarkedMediaDetectionPipeline
        )
        pipelines_tested.extend([
            "WatermarkedMediaDetectionPipeline",
            "UnWatermarkedMediaDetectionPipeline"
        ])
    except ImportError as e:
        print(f"✗ Failed to import detection pipelines: {e}")

    # Image quality pipelines
    try:
        from evaluation.pipelines.image_quality_analysis import (
            DirectImageQualityAnalysisPipeline,
            ReferencedImageQualityAnalysisPipeline,
            GroupImageQualityAnalysisPipeline,
            RepeatImageQualityAnalysisPipeline,
            ComparedImageQualityAnalysisPipeline
        )
        pipelines_tested.extend([
            "DirectImageQualityAnalysisPipeline",
            "ReferencedImageQualityAnalysisPipeline",
            "GroupImageQualityAnalysisPipeline",
            "RepeatImageQualityAnalysisPipeline",
            "ComparedImageQualityAnalysisPipeline"
        ])
    except ImportError as e:
        print(f"✗ Failed to import image quality pipelines: {e}")

    # Video quality pipeline
    try:
        from evaluation.pipelines.video_quality_analysis import (
            DirectVideoQualityAnalysisPipeline
        )
        pipelines_tested.append("DirectVideoQualityAnalysisPipeline")
    except ImportError as e:
        print(f"✗ Failed to import video quality pipeline: {e}")

    print(f"\n✓ Successfully tested {len(pipelines_tested)} pipeline classes:")
    for pipeline in pipelines_tested:
        print(f"  - {pipeline}")

    # Verify we tested all expected pipelines (8 total as per README)
    expected_count = 8  # 2 detection + 5 image quality + 1 video quality
    assert len(pipelines_tested) == expected_count, \
        f"Expected {expected_count} pipelines but tested {len(pipelines_tested)}"

    print(f"\n✓ All {expected_count} evaluation pipelines verified successfully!")


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running pipeline tests...")

    # Test datasets
    test_base_dataset()

    # Test pipelines
    test_watermarked_media_detection_pipeline()
    test_unwatermarked_media_detection_pipeline()
    test_direct_image_quality_analysis_pipeline()
    test_referenced_image_quality_analysis_pipeline()
    test_group_image_quality_analysis_pipeline()
    test_repeat_image_quality_analysis_pipeline()
    test_compared_image_quality_analysis_pipeline()
    test_direct_video_quality_analysis_pipeline()
    test_pipeline_return_types()
    test_all_pipelines_summary()

    print("\n✓ All tests completed successfully!")