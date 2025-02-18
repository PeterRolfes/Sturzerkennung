import h5py
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

class FallDetectionDataset(Dataset):
    """Dataset class for fall detection using both frame and video features"""
    def __init__(self, frames_file='frames_data.h5', videos_file='videos_data.h5'):
        # Load video features
        with h5py.File(videos_file, 'r') as data:
            self.floorline_masks = data['floorline_masks'][:]
            self.last_avg_frames = data['last_avg_frames'][:]
            self.movement_frames = data['movement_frames'][:]
            self.labels = data['labels'][:]

        # Store frames file path for on-demand loading
        self.frames_file = frames_file
        
        # Load video IDs for frame mapping
        with h5py.File(frames_file, 'r') as f:
            self.video_ids = np.array(f['videoIDs'][:])
    
    def get_video_label(self, video_id):
        """Get the label for a specific video"""
        # Find the index where this video_id appears in the dataset
        
        return int(self.labels[video_id])
        
    def get_frames_sequence(self, video_id):
        """Get all frames and corresponding floorline mask for a specific video"""
        with h5py.File(self.frames_file, 'r') as f:
            frames = np.array(f['frames'][:])
            indices = np.where(self.video_ids == video_id)[0]
            return frames[indices], self.floorline_masks[video_id]
    
    def __getitem__(self, idx):
        # Get video features
        floorline_mask = self.floorline_masks[idx]
        last_avg_frame = self.last_avg_frames[idx]
        movement_frame = self.movement_frames[idx]
        
        # Stack features
        features = torch.stack([
            torch.tensor(last_avg_frame, dtype=torch.float32),
            torch.tensor(movement_frame, dtype=torch.float32),
            torch.tensor(floorline_mask, dtype=torch.float32)
        ], dim=0)
        
        # Normalize
        features = features / 255.0
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long).squeeze()
        
        return features, label
    
    def __len__(self):
        return len(self.labels)


class WarmPointAnalyzer:
    """Analyzes ratio of largest warm components above/below floor line"""
    def __init__(self, temp_threshold=150):
        self.temp_threshold = temp_threshold
    
    def get_largest_component(self, mask):
        """Find largest connected component in a binary mask"""
        if not np.any(mask):
            return np.zeros_like(mask), 0
            
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        if num_labels <= 1:  # Only background
            return np.zeros_like(mask), 0
            
        # Find largest component
        component_sizes = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        largest_idx = np.argmax(component_sizes) + 1  # Add 1 because we skipped background
        
        # Create mask for largest component
        largest_mask = labels == largest_idx
        largest_size = component_sizes[largest_idx - 1]
        
        return largest_mask, largest_size
    
    def analyze_frame_regions(self, frame, floorline_mask):
        """
        Analyze largest warm regions above and below floor line
        
        Args:
            frame (np.ndarray): Input frame (assumed to be in range 0-255)
            floorline_mask (np.ndarray): Binary mask indicating floor/body line
            
        Returns:
            dict: Dictionary containing:
                - below_mask: Binary mask of largest component below floor line
                - above_mask: Binary mask of largest component above floor line
                - below_size: Size of largest component below floor line
                - above_size: Size of largest component above floor line
                - ratio: Ratio of below/above sizes
        """
        # Ensure frame is in correct range
        if frame.max() <= 1.0:
            frame = frame * 255.0
            
        # Create binary mask of all warm points
        warm_mask = frame > self.temp_threshold
        
        # Split into below and above regions
        below_region = warm_mask & (floorline_mask > 0)
        above_region = warm_mask & (floorline_mask == 0)
        
        # Get largest components in each region
        below_mask, below_size = self.get_largest_component(below_region)
        above_mask, above_size = self.get_largest_component(above_region)
        
        # Calculate ratio (handle division by zero)
        ratio = below_size / above_size if above_size > 0 else float('inf')
        
        return {
            'below_mask': below_mask,
            'above_mask': above_mask,
            'below_size': below_size,
            'above_size': above_size,
            'ratio': ratio
        }
    
    def analyze_frame_sequence(self, frames, floorline_mask):
        """Analyze ratio changes in sequence of frames"""
        results = []
        prev_ratio = None
        
        for frame in frames:
            frame_analysis = self.analyze_frame_regions(frame, floorline_mask)
            
            # Store absolute ratio value rather than change
            frame_analysis['ratio_value'] = frame_analysis['ratio']
            results.append(frame_analysis)
                
        return results
    
    def get_sequence_statistics(self, sequence_results):
        """Calculate statistics over a sequence of frame analyses"""
        ratios = [r['ratio_value'] for r in sequence_results]
        finite_ratios = [r for r in ratios if not np.isinf(r)]
        
        if not finite_ratios:
            return {
                'max_ratio_diff': 0,
                'max_diff_frame_idx': 0,
                'min_ratio': 0,
                'max_ratio': 0
            }
        
        # Find minimum and maximum ratios
        min_ratio = min(finite_ratios)
        max_ratio = max(finite_ratios)
        
        # Calculate maximum absolute difference between ratios
        max_ratio_diff = abs(max_ratio - min_ratio)
        
        # Find frame indices for visualization
        min_ratio_frame = ratios.index(min_ratio)
        max_ratio_frame = ratios.index(max_ratio)
        
        # Use the frame with the larger absolute ratio for visualization
        max_diff_frame_idx = max_ratio_frame if abs(max_ratio) > abs(min_ratio) else min_ratio_frame
        
        return {
            'max_ratio_diff': max_ratio_diff,
            'max_diff_frame_idx': max_diff_frame_idx,
            'min_ratio': min_ratio,
            'max_ratio': max_ratio
        }
    
    def visualize_min_max_frames(self, frames, floorline_mask, frame_results):
        """Create side-by-side visualization of frames with min and max ratio values"""
        # Get ratios from results
        ratios = [r['ratio_value'] for r in frame_results]
        finite_ratios = [r for r in ratios if not np.isinf(r)]
        
        if not finite_ratios:
            print("No valid ratios found for visualization")
            return
            
        min_ratio = min(finite_ratios)
        max_ratio = max(finite_ratios)
        min_frame_idx = ratios.index(min_ratio)
        max_frame_idx = ratios.index(max_ratio)
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Helper function to create visualization for a single frame
        def create_frame_visualization(frame, floorline_mask, below_mask, above_mask):
            vis_img = np.zeros((*frame.shape, 3), dtype=np.uint8)
            
            # Convert frame to grayscale for background
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            vis_img[..., 0] = frame  # Red channel
            vis_img[..., 1] = frame  # Green channel
            vis_img[..., 2] = frame  # Blue channel
            
            # Draw floor line boundary in blue
            floor_boundary = cv2.findContours(
                floorline_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )[0]
            cv2.drawContours(vis_img, floor_boundary, -1, (0, 0, 255), 1)
            
            # Mark largest component below floor line in red
            vis_img[below_mask, 0] = 255  # Red channel
            vis_img[below_mask, 1] = 0    # Green channel
            vis_img[below_mask, 2] = 0    # Blue channel
            
            # Mark largest component above floor line in green
            vis_img[above_mask, 0] = 0    # Red channel
            vis_img[above_mask, 1] = 255  # Green channel
            vis_img[above_mask, 2] = 0    # Blue channel
            
            return vis_img
        
        # Create and display frame with minimum ratio
        min_result = frame_results[min_frame_idx]
        min_vis = create_frame_visualization(
            frames[min_frame_idx],
            floorline_mask,
            min_result['below_mask'],
            min_result['above_mask']
        )
        ax1.imshow(min_vis)
        ax1.set_title(f'Frame {min_frame_idx}\nMinimum Ratio: {min_ratio:.2f}\n'
                    f'Below Size: {min_result["below_size"]}, Above Size: {min_result["above_size"]}')
        ax1.axis('off')
        
        # Create and display frame with maximum ratio
        max_result = frame_results[max_frame_idx]
        max_vis = create_frame_visualization(
            frames[max_frame_idx],
            floorline_mask,
            max_result['below_mask'],
            max_result['above_mask']
        )
        ax2.imshow(max_vis)
        ax2.set_title(f'Frame {max_frame_idx}\nMaximum Ratio: {max_ratio:.2f}\n'
                    f'Below Size: {max_result["below_size"]}, Above Size: {max_result["above_size"]}\n'
                    f'Max Ratio Difference: {abs(max_ratio - min_ratio):.2f}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

def analyze_video_warm_points(video_id, visualize=False, analyzer=None, temp_threshold=170):
    """Analyze changes in ratio of largest warm regions"""
    # Initialize dataset and analyzer if not provided
    dataset = FallDetectionDataset()
    if analyzer is None:
        analyzer = WarmPointAnalyzer(temp_threshold=temp_threshold)
    
    # Get frame sequence and floorline mask
    frames, floorline_mask = dataset.get_frames_sequence(video_id)
    
    # Analyze frames
    frame_results = analyzer.analyze_frame_sequence(frames, floorline_mask)
    
    # Create visualization of min and max frames
    if visualize:
        analyzer.visualize_min_max_frames(frames, floorline_mask, frame_results)
    
    # Get sequence statistics
    sequence_stats = analyzer.get_sequence_statistics(frame_results)
    
    # Get the video label
    label = dataset.get_video_label(video_id)
    sequence_stats['label'] = label
    
    return sequence_stats


# Add these new functions to your existing code

def classify_fall(max_ratio_diff, ratio_diff_threshold):
    """
    Classify a sequence as fall/no-fall based on maximum ratio difference
    
    Args:
        max_ratio_diff: Maximum absolute difference in warm points ratio
        ratio_diff_threshold: Threshold for classification
        
    Returns:
        1 if classified as fall, 0 otherwise
    """
    # If the maximum ratio difference exceeds threshold, classify as fall
    return 1 if max_ratio_diff > ratio_diff_threshold else 0


def evaluate_classifier_with_params(ratio_threshold, temp_threshold, visualize=False):
    """
    Evaluate classifier performance with specific temperature and ratio thresholds
    """
    y_true = []
    y_pred = []
    
    dataset = FallDetectionDataset()
    analyzer = WarmPointAnalyzer(temp_threshold=temp_threshold)
    
    for video_id in range(0, 780):
        if video_id % 50 == 0:
            print(f"Evaluated {video_id} out of 780 videos")

        # Get sequence statistics and true label
        stats = analyze_video_warm_points(video_id, visualize=False, analyzer=analyzer)
        
        # Make prediction based on maximum ratio difference
        prediction = classify_fall(stats['max_ratio_diff'], ratio_threshold)
        
        y_true.append(stats['label'])
        y_pred.append(prediction)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    return metrics

def find_optimal_parameters(ratio_thresholds=None, temp_thresholds=None):
    """Find optimal thresholds by evaluating multiple combinations"""
    if ratio_thresholds is None:
        ratio_thresholds = np.arange(0.5, 2.5, 0.1)
    if temp_thresholds is None:
        temp_thresholds = np.arange(100, 200, 10)
    
    best_metrics = None
    best_params = None
    best_f1 = -1
    
    results = []
    
    for temp_threshold in temp_thresholds:
        print(f"\nTesting temperature threshold: {temp_threshold}")
        for ratio_threshold in ratio_thresholds:
            print(f"  Ratio threshold: {ratio_threshold:.2f}")
            metrics = evaluate_classifier_with_params(ratio_threshold, temp_threshold, visualize=False)
            results.append({
                'temp_threshold': temp_threshold,
                'ratio_threshold': ratio_threshold,
                **metrics
            })
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_metrics = metrics
                best_params = {
                    'temp_threshold': temp_threshold,
                    'ratio_threshold': ratio_threshold
                }
    
    # Visualize results
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 8))
    pivot_table = results_df.pivot(
        index='temp_threshold', 
        columns='ratio_threshold', 
        values='f1'
    )
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
    plt.title('F1 Score by Temperature and Ratio Thresholds')
    plt.xlabel('Ratio Threshold')
    plt.ylabel('Temperature Threshold')
    plt.show()
    
    return {
        'parameters': best_params,
        'metrics': best_metrics
    }

def main():
    """Main function to demonstrate usage"""
    grid_search = False
    final_eval = False
    visualize_frame = False

    # Default parameters (best results from grid search)
    best_ratio_threshold =  1.40
    best_temp_threshold = 170

    if grid_search:
        print("Finding optimal parameters...")
        result = find_optimal_parameters()
        best_params = result['parameters']
        print(f"\nBest parameters:")
        print(f"Temperature threshold: {best_params['temp_threshold']}")
        print(f"Ratio threshold: {best_params['ratio_threshold']:.2f}")

    if final_eval:
        print("\nFinal evaluation with best parameters:")
        metrics = evaluate_classifier_with_params(
            best_ratio_threshold, 
            best_temp_threshold, 
            visualize=True
        )

        print("\nFinal metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

    if visualize_frame:
        video_id = 620
        stats = analyze_video_warm_points(video_id, visualize=True, temp_threshold=best_temp_threshold)
        prediction = classify_fall(stats['max_ratio_diff'], best_ratio_threshold)
        print(f"\nVideo {video_id}:")
        print(f"Maximum ratio difference: {stats['max_ratio_diff']:.2f}")
        print(f"Min ratio: {stats['min_ratio']:.2f}, Max ratio: {stats['max_ratio']:.2f}")
        print(f"True label: {'Fall' if stats['label'] == 1 else 'No Fall'}")
        print(f"Prediction: {'Fall' if prediction == 1 else 'No Fall'}")
        print(f"Correct: {'Yes' if prediction == stats['label'] else 'No'}")    

if __name__ == "__main__":
    main()