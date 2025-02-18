# Thermal Fall Detection System

This project implements a fall detection system using thermal imaging data. The system analyzes sequences of thermal frames to detect falls by tracking the distribution of warm regions above and below a detected floor line.

## Overview

The system works by analyzing the ratio of warm regions in thermal images, particularly focusing on how these regions are distributed relative to a detected floor line. A fall is characterized by significant changes in this distribution over time.

## Key Components

### FallDetectionDataset

A PyTorch Dataset class that handles loading and processing of thermal video data. It manages:
- Frame sequences from HDF5 files
- Floor line masks
- Movement frames
- Video labels (fall/no fall)

### WarmPointAnalyzer

The core analysis class that implements the fall detection algorithm. Key features:

- Identifies and measures warm regions in thermal frames
- Analyzes the distribution of warm regions above and below the floor line
- Tracks changes in warm region ratios over time
- Provides visualization tools for analysis

Key methods:
- `analyze_frame_regions`: Analyzes single frames
- `analyze_frame_sequence`: Processes sequences of frames
- `get_sequence_statistics`: Calculates sequence-level statistics
- `visualize_min_max_frames`: Creates visualizations for analysis

### Classification System

The system includes functions for:
- Binary classification of falls based on ratio thresholds
- Parameter optimization through grid search
- Performance evaluation using standard metrics

## How It Works

1. **Thermal Frame Processing**
   - Each frame is processed to identify regions above a temperature threshold
   - The largest connected components above and below the floor line are identified

2. **Ratio Analysis**
   - The system calculates the ratio of warm regions below/above the floor line
   - Significant changes in this ratio over time indicate potential falls

3. **Classification**
   - Falls are detected when the maximum ratio difference exceeds a threshold
   - Optimal thresholds are determined through grid search

## Performance Metrics

The system evaluates performance using:
- Accuracy
- Precision
- Recall
- F1 Score

## Usage

### Basic Usage

```python
# Analyze a single video
stats = analyze_video_warm_points(video_id, visualize=True)

# Evaluate classifier with specific parameters
metrics = evaluate_classifier_with_params(
    ratio_threshold=1.40,
    temp_threshold=170
)
```

### Parameter Optimization

```python
# Find optimal parameters through grid search
result = find_optimal_parameters()
```

### Visualization

The system includes visualization tools to help understand the detection process:
- Side-by-side comparison of frames with minimum and maximum ratios
- Heatmaps of classifier performance across different parameters
- Visual markers for detected warm regions and floor lines

## Default Parameters

The system comes with pre-optimized default parameters:
- Temperature threshold: 170
- Ratio threshold: 1.40

These values were determined through grid search optimization.

## Dependencies

- PyTorch
- NumPy
- OpenCV
- h5py
- matplotlib
- seaborn
- pandas
- scikit-learn

## Data Format

The system expects data in HDF5 format with the following structure:
- `frames_data.h5`: Contains frame sequences and video IDs
- `videos_data.h5`: Contains floor line masks, movement frames, and labels

## Notes

- The system is designed for thermal image sequences
- Performance depends on accurate floor line detection
- Visualization tools are provided for debugging and analysis
- Grid search can be used to optimize parameters for specific environments
