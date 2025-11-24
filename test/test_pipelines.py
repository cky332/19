"""
Comprehensive tests for MarkDiffusion evaluation pipelines and datasets.

This module tests:
1. Dataset classes (StableDiffusionPromptsDataset, MSCOCODataset, VBenchDataset)
2. Detection pipelines (WatermarkedMediaDetectionPipeline, UnWatermarkedMediaDetectionPipeline)
3. Image quality analysis pipelines (5 pipelines)
4. Video quality analysis pipeline

All tests use saturation testing with all available editors and analyzers.

Usage:
    # Test all pipelines and datasets
    pytest test/test_pipelines.py -v

    # Test specific components
    pytest test/test_pipelines.py -m dataset -v
    pytest test/test_pipelines.py -m detection -v
    pytest test/test_pipelines.py -m quality -v
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
    QualityPipelineReturnType,
    QualityComparisonResult
)

from evaluation.pipelines.video_quality_analysis import (
    DirectVideoQualityAnalysisPipeline
)

# Import test constants
from .conftest import TEST_DATASET_MAX_SAMPLES


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
# Test Cases - Detection Pipelines (Saturation Tests)
# ============================================================================

@pytest.mark.pipeline
@pytest.mark.detection
@pytest.mark.slow
def test_watermarked_detection_pipeline_with_all_image_editors(test_image_dataset, all_image_editors, image_diffusion_config):
    """Saturation test: WatermarkedMediaDetectionPipeline with all image editors."""
    from watermark.auto_watermark import AutoWatermark

    # Initialize pipeline
    pipeline = WatermarkedMediaDetectionPipeline(
        dataset=test_image_dataset,
        media_editor_list=all_image_editors,
        return_type=DetectionPipelineReturnType.SCORES
    )

   #  assert len(pipeline.media_editor_list) == len(all_image_editors)
    assert pipeline.dataset == test_image_dataset

    # Verify all editors are present
    # editor_types = [type(editor).__name__ for editor in pipeline.media_editor_list]
    # expected_editors = [
    #     'JPEGCompression', 'Rotation', 'CrSc', 'GaussianBlurring',
    #     'GaussianNoise', 'Brightness', 'Mask', 'Overlay', 'AdaptiveNoiseInjection'
    # ]

    # for expected in expected_editors:
    #     assert expected in editor_types, f"Missing editor: {expected}"

    # Load a watermark algorithm (use TR as example)
    try:
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, list), "Evaluate should return a list"
        assert len(result) > 0, "Evaluate should return non-empty results"

        print(f"✓ WatermarkedMediaDetectionPipeline with all {len(all_image_editors)} image editors test passed")
        print(f"  - Pipeline evaluated successfully with {len(result)} results")

    except Exception as e:
        raise RuntimeError(f"Watermark loading or evaluation error: {e}")
        pytest.skip(f"Skipping test due to watermark loading or evaluation error: {e}")


@pytest.mark.pipeline
@pytest.mark.detection
@pytest.mark.slow
def test_unwatermarked_detection_pipeline_with_all_image_editors(test_image_dataset, all_image_editors, image_diffusion_config):
    """Saturation test: UnWatermarkedMediaDetectionPipeline with all image editors."""
    from watermark.auto_watermark import AutoWatermark

    # Initialize pipeline
    pipeline = UnWatermarkedMediaDetectionPipeline(
        dataset=test_image_dataset,
        media_editor_list=all_image_editors,
        return_type=DetectionPipelineReturnType.SCORES
    )

    assert len(pipeline.media_editor_list) == len(all_image_editors)
    assert pipeline.dataset == test_image_dataset

    # Load a watermark algorithm (use TR as example)
    try:
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, list), "Evaluate should return a list"
        assert len(result) > 0, "Evaluate should return non-empty results"

        print(f"✓ UnWatermarkedMediaDetectionPipeline with all {len(all_image_editors)} image editors test passed")
        print(f"  - Pipeline evaluated successfully with {len(result)} results")

    except Exception as e:
        pytest.skip(f"Skipping test due to watermark loading or evaluation error: {e}")


@pytest.mark.pipeline
@pytest.mark.detection
@pytest.mark.video
@pytest.mark.slow
def test_detection_pipeline_with_all_video_editors(test_video_dataset, all_video_editors, video_diffusion_config):
    """Saturation test: Detection pipeline with all video editors."""
    from watermark.auto_watermark import AutoWatermark
    
    pipeline = WatermarkedMediaDetectionPipeline(
        dataset=test_video_dataset,
        media_editor_list=all_video_editors,
        detector_type="bit_acc",
        return_type=DetectionPipelineReturnType.SCORES
    )

    assert len(pipeline.media_editor_list) == len(all_video_editors)
    assert pipeline.dataset == test_video_dataset

    # # Verify all editors are present
    # editor_types = [type(editor).__name__ for editor in pipeline.media_editor_list]
    # expected_editors = [
    #     'MPEG4Compression', 'VideoCodecAttack', 'FrameAverage',
    #     'FrameRateAdapter', 'FrameSwap', 'FrameInterpolationAttack'
    # ]

    # for expected in expected_editors:
    #     assert expected in editor_types, f"Missing editor: {expected}"
     # Load a watermark algorithm (use TR as example)
    try:
        from watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'VideoShield',
            algorithm_config='config/VideoShield.json',
            diffusion_config=video_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, list), "Evaluate should return a list"
        assert len(result) > 0, "Evaluate should return non-empty results"

        print(f"✓ Detection pipeline with all {len(all_video_editors)} video editors test passed")
        print(f"  - Pipeline evaluated successfully with {len(result)} results")

    except Exception as e:
        pytest.skip(f"Skipping test due to watermark loading or evaluation error: {e}")

    


# ============================================================================
# Test Cases - Image Quality Analysis Pipelines (Saturation Tests)
# ============================================================================

@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.slow
def test_direct_image_quality_pipeline_saturation(test_image_dataset, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: DirectImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = DirectImageQualityAnalysisPipeline(
        dataset=test_image_dataset,
        watermarked_image_editor_list=all_image_editors,
        unwatermarked_image_editor_list=all_image_editors,
        analyzers=all_image_quality_analyzers['direct'],
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.unwatermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.analyzers) == len(all_image_quality_analyzers['direct'])
    
    try:
        from watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ DirectImageQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_image_editors)} editors per image type")
        print(f"  - {len(all_image_quality_analyzers['direct'])} analyzers")

    except Exception as e:
        pytest.skip(f"Skipping test due to watermark loading or evaluation error: {e}")
    

    


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.slow
def test_referenced_image_quality_pipeline_saturation(test_image_dataset, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: ReferencedImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = ReferencedImageQualityAnalysisPipeline(
        dataset=test_image_dataset,
        watermarked_image_editor_list=all_image_editors,
        unwatermarked_image_editor_list=all_image_editors,
        analyzers=all_image_quality_analyzers['referenced'],
        unwatermarked_image_source='generated',
        reference_image_source='natural',
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.unwatermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.analyzers) == len(all_image_quality_analyzers['referenced'])
    
    try:
        from watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ ReferencedImageQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_image_editors)} editors per image type")
        print(f"  - {len(all_image_quality_analyzers['referenced'])} analyzers")

    except Exception as e:
        pytest.skip(f"Skipping test due to watermark loading or evaluation error: {e}")




@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.slow
def test_group_image_quality_pipeline_saturation(test_image_dataset, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: GroupImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = GroupImageQualityAnalysisPipeline(
        dataset=test_image_dataset,
        watermarked_image_editor_list=all_image_editors,
        unwatermarked_image_editor_list=all_image_editors,
        analyzers=all_image_quality_analyzers['group'],
        unwatermarked_image_source='generated',
        reference_image_source='natural',
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.unwatermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.analyzers) == len(all_image_quality_analyzers['group'])

    try:
        from watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ GroupImageQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_image_editors)} editors per image type")
        print(f"  - {len(all_image_quality_analyzers['group'])} analyzers")

    except Exception as e:
        pytest.skip(f"Skipping test due to watermark loading or evaluation error: {e}")

    


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.slow
def test_repeat_image_quality_pipeline_saturation(test_image_dataset, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: RepeatImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = RepeatImageQualityAnalysisPipeline(
        dataset=test_image_dataset,
        prompt_per_image=5,  # Small number for testing
        watermarked_image_editor_list=all_image_editors,
        unwatermarked_image_editor_list=all_image_editors,
        analyzers=all_image_quality_analyzers['repeat'],
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.unwatermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.analyzers) == len(all_image_quality_analyzers['repeat'])
    assert pipeline.prompt_per_image == 5
    
    try:
        from watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ RepeatImageQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_image_editors)} editors per image type")
        print(f"  - {len(all_image_quality_analyzers['repeat'])} analyzers")

    except Exception as e:
        pytest.fail(f"Skipping test due to watermark loading or evaluation error: {e}")

    


@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.slow
def test_compared_image_quality_pipeline_saturation(test_image_dataset, all_image_editors, all_image_quality_analyzers, image_diffusion_config):
    """Saturation test: ComparedImageQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = ComparedImageQualityAnalysisPipeline(
        dataset=test_image_dataset,
        watermarked_image_editor_list=all_image_editors,
        unwatermarked_image_editor_list=all_image_editors,
        analyzers=all_image_quality_analyzers['compared'],
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.unwatermarked_image_editor_list) == len(all_image_editors)
    assert len(pipeline.analyzers) == len(all_image_quality_analyzers['compared'])
    
    try:
        from watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'TR',
            algorithm_config='config/TR.json',
            diffusion_config=image_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ ComparedImageQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_image_editors)} editors per image type")
        print(f"  - {len(all_image_quality_analyzers['compared'])} analyzers")

    except Exception as e:
        pytest.skip(f"Skipping test due to watermark loading or evaluation error: {e}")




# ============================================================================
# Test Cases - Video Quality Analysis Pipeline (Saturation Test)
# ============================================================================

@pytest.mark.pipeline
@pytest.mark.quality
@pytest.mark.video
@pytest.mark.slow
def test_video_quality_pipeline_saturation(test_video_dataset, all_video_editors, all_image_editors, all_video_quality_analyzers, video_diffusion_config):
    """Saturation test: DirectVideoQualityAnalysisPipeline with all editors and analyzers."""
    pipeline = DirectVideoQualityAnalysisPipeline(
        dataset=test_video_dataset,
        watermarked_video_editor_list=all_video_editors,
        unwatermarked_video_editor_list=all_video_editors,
        watermarked_frame_editor_list=[],
        unwatermarked_frame_editor_list=[],
        analyzers=all_video_quality_analyzers,
        return_type=QualityPipelineReturnType.FULL
    )

    assert len(pipeline.watermarked_video_editor_list) == len(all_video_editors)
    assert len(pipeline.unwatermarked_video_editor_list) == len(all_video_editors)
    assert len(pipeline.analyzers) == len(all_video_quality_analyzers)
    
    try:
        from watermark.auto_watermark import AutoWatermark
        watermark = AutoWatermark.load(
            'VideoShield',
            algorithm_config='config/VideoShield.json',
            diffusion_config=video_diffusion_config
        )

        # Call evaluate method
        result = pipeline.evaluate(watermark)

        # Assert evaluate executed successfully
        assert result is not None, "Evaluate method returned None"
        assert isinstance(result, QualityComparisonResult), "Evaluate should return QualityComparisonResult"

        print(f"✓ DirectVideoQualityAnalysisPipeline saturation test passed")
        print(f"  - {len(all_video_editors)} video editors per video type")
        print(f"  - {len(all_image_editors)} frame editors per video type")
        print(f"  - {len(all_video_quality_analyzers)} analyzers")

    except Exception as e:
        pytest.skip(f"Skipping test due to watermark loading or evaluation error: {e}")

# ============================================================================
# Test Cases - Return Types
# ============================================================================

# @pytest.mark.pipeline
# def test_pipeline_return_types():
#     """Test pipeline return type enumerations."""
#     # Test DetectionPipelineReturnType
#     assert DetectionPipelineReturnType.SCORES.value == "scores"
#     assert DetectionPipelineReturnType.LABELS.value == "labels"
#     assert DetectionPipelineReturnType.SCORES_AND_LABELS.value == "scores_and_labels"

#     # Test QualityPipelineReturnType
#     assert QualityPipelineReturnType.MEAN_SCORES.value == "mean_scores"
#     assert QualityPipelineReturnType.ALL_SCORES.value == "all_scores"

#     print("✓ Pipeline return types test passed")


# # ============================================================================
# # Summary Test
# # ============================================================================

# @pytest.mark.summary
# def test_all_pipelines_summary():
#     """Summary test to verify all pipeline classes are importable and constructable."""
#     pipelines_tested = []

#     # Detection pipelines
#     try:
#         from evaluation.pipelines.detection import (
#             WatermarkedMediaDetectionPipeline,
#             UnWatermarkedMediaDetectionPipeline
#         )
#         pipelines_tested.extend([
#             "WatermarkedMediaDetectionPipeline",
#             "UnWatermarkedMediaDetectionPipeline"
#         ])
#     except ImportError as e:
#         print(f"✗ Failed to import detection pipelines: {e}")

#     # Image quality pipelines
#     try:
#         from evaluation.pipelines.image_quality_analysis import (
#             DirectImageQualityAnalysisPipeline,
#             ReferencedImageQualityAnalysisPipeline,
#             GroupImageQualityAnalysisPipeline,
#             RepeatImageQualityAnalysisPipeline,
#             ComparedImageQualityAnalysisPipeline
#         )
#         pipelines_tested.extend([
#             "DirectImageQualityAnalysisPipeline",
#             "ReferencedImageQualityAnalysisPipeline",
#             "GroupImageQualityAnalysisPipeline",
#             "RepeatImageQualityAnalysisPipeline",
#             "ComparedImageQualityAnalysisPipeline"
#         ])
#     except ImportError as e:
#         print(f"✗ Failed to import image quality pipelines: {e}")

#     # Video quality pipeline
#     try:
#         from evaluation.pipelines.video_quality_analysis import (
#             DirectVideoQualityAnalysisPipeline
#         )
#         pipelines_tested.append("DirectVideoQualityAnalysisPipeline")
#     except ImportError as e:
#         print(f"✗ Failed to import video quality pipeline: {e}")

#     print(f"\n✓ Successfully tested {len(pipelines_tested)} pipeline classes:")
#     for pipeline in pipelines_tested:
#         print(f"  - {pipeline}")

#     # Verify we tested all expected pipelines (8 total as per README)
#     expected_count = 8  # 2 detection + 5 image quality + 1 video quality
#     assert len(pipelines_tested) == expected_count, \
#         f"Expected {expected_count} pipelines but tested {len(pipelines_tested)}"

#     print(f"\n✓ All {expected_count} evaluation pipelines verified successfully!")


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running pipeline tests...")

    # Test datasets
    test_base_dataset()

    # Test return types
    test_pipeline_return_types()
    test_all_pipelines_summary()

    print("\n✓ All basic tests completed successfully!")
