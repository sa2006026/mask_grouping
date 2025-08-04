"""Droplet-only mask processor for basic SAM segmentation with filtering."""

import os
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
from segment_anything import SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DropletProcessor:
    """Simple droplet processor that applies basic SAM masking and filtering."""
    
    def __init__(self, mask_generator, min_circularity=0.53, max_blob_distance=50):
        """
        Initialize droplet processor.
        
        Args:
            mask_generator: SAM mask generator with crop_n_layers=1
            min_circularity: Minimum circularity threshold
            max_blob_distance: Maximum blob distance threshold
        """
        self.mask_generator = mask_generator
        self.min_circularity = min_circularity
        self.max_blob_distance = max_blob_distance
    
    def process_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image array through basic SAM masking and filtering.
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary containing results and statistics
        """
        print(f"Processing image in droplet mode: {image.shape}")
        
        # Step 1: Generate masks using SAM with crop_n_layers=1
        print("Generating masks with basic SAM (crop_n_layers=1)...")
        all_masks = self.mask_generator.generate(image)
        print(f"Generated {len(all_masks)} masks")
        
        # Add original SAM indices to each mask for tracking
        for i, mask in enumerate(all_masks):
            mask['original_sam_id'] = i
            
        # Step 2: Add circularity information
        all_masks = self._add_circularity_to_masks(all_masks)
        
        # Step 3: Filter out edge-touching masks
        edge_filtered_masks, removed_edge = self._filter_problematic_masks(all_masks, image.shape)
        
        # Step 4: Filter by circularity
        circularity_filtered_masks, removed_circularity = self._filter_by_circularity(
            edge_filtered_masks, self.min_circularity
        )
        
        # Step 5: Filter by blob distance
        filtered_masks, removed_multi_blob = self._filter_by_blob_distance(
            circularity_filtered_masks, self.max_blob_distance
        )
        
        # Step 6: Remove duplicate masks (similar to original but simpler)
        final_masks, removed_duplicates = self._remove_duplicate_masks(filtered_masks)
        
        # Step 7: Calculate statistics
        statistics = self._calculate_droplet_statistics(final_masks)
        
        return {
            'total_masks': len(all_masks),
            'filtered_masks': len(filtered_masks),
            'final_masks': final_masks,
            'statistics': statistics,
            'processing_details': {
                'removed_edge': len(removed_edge),
                'removed_circularity': len(removed_circularity),
                'removed_multi_blob': len(removed_multi_blob),
                'removed_duplicates': len(removed_duplicates)
            }
        }
    
    def _add_circularity_to_masks(self, masks: List[Dict]) -> List[Dict]:
        """Add circularity information to all masks."""
        print("Calculating circularity for all masks...")
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            circularity = self._calculate_circularity(mask)
            mask_data['circularity'] = circularity
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(masks)} masks...")
        
        print(f"Circularity calculation complete for {len(masks)} masks")
        return masks
    
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """Calculate circularity of a mask (4π*area/perimeter²)."""
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate area and perimeter
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularity = 4π * area / perimeter²
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        return min(circularity, 1.0)  # Cap at 1.0
    
    def _filter_problematic_masks(self, masks: List[Dict], image_shape: Tuple) -> Tuple[List[Dict], List[Dict]]:
        """Filter out edge-touching masks."""
        print("Filtering out edge-touching masks...")
        
        filtered_masks = []
        removed_edge = []
        edge_threshold = 5
        
        for mask_data in masks:
            bbox = mask_data['bbox']
            
            # Check if mask is touching edges
            is_touching_edge = self._is_mask_touching_edge(bbox, image_shape, edge_threshold)
            
            if is_touching_edge:
                removed_edge.append(mask_data)
            else:
                filtered_masks.append(mask_data)
        
        print(f"Removed {len(removed_edge)} edge-touching masks, {len(filtered_masks)} remaining")
        return filtered_masks, removed_edge
    
    def _is_mask_touching_edge(self, bbox: List, image_shape: Tuple, edge_threshold: int = 5) -> bool:
        """Check if mask is touching the image edges."""
        height, width = image_shape[:2]
        x, y, w, h = bbox
        
        # Check if bounding box is close to edges
        touching_left = x <= edge_threshold
        touching_right = (x + w) >= (width - edge_threshold)
        touching_top = y <= edge_threshold
        touching_bottom = (y + h) >= (height - edge_threshold)
        
        return touching_left or touching_right or touching_top or touching_bottom
    
    def _filter_by_circularity(self, masks: List[Dict], min_circularity: float) -> Tuple[List[Dict], List[Dict]]:
        """Filter masks by circularity threshold."""
        print(f"Filtering masks by circularity (min_circularity: {min_circularity})...")
        
        filtered_masks = []
        removed_circularity = []
        
        for mask_data in masks:
            circularity = mask_data.get('circularity', 0.0)
            
            if circularity >= min_circularity:
                filtered_masks.append(mask_data)
            else:
                removed_circularity.append(mask_data)
        
        print(f"Removed {len(removed_circularity)} low circularity masks, {len(filtered_masks)} remaining")
        return filtered_masks, removed_circularity
    
    def _filter_by_blob_distance(self, masks: List[Dict], max_distance: float) -> Tuple[List[Dict], List[Dict]]:
        """Filter out masks where blobs are separated by more than max_distance pixels."""
        print(f"Filtering masks by blob distance (max_distance: {max_distance} pixels)...")
        
        filtered_masks = []
        removed_multi_blob = []
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            blob_distance = self._calculate_blob_distance(mask)
            
            mask_data['blob_distance'] = blob_distance
            
            if blob_distance <= max_distance:
                filtered_masks.append(mask_data)
            else:
                removed_multi_blob.append(mask_data)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(masks)} masks...")
        
        print(f"Removed {len(removed_multi_blob)} distant-blob masks, {len(filtered_masks)} remaining")
        return filtered_masks, removed_multi_blob
    
    def _calculate_blob_distance(self, mask: np.ndarray) -> float:
        """Calculate maximum distance between connected components (blobs) in a mask."""
        # Convert mask to uint8
        mask_uint8 = mask.astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        if num_labels <= 2:  # Only background and one component
            return 0.0
        
        # Calculate distances between all pairs of components
        max_distance = 0.0
        for i in range(1, num_labels):  # Skip background (label 0)
            for j in range(i + 1, num_labels):
                # Get centroids
                cx1, cy1 = centroids[i]
                cx2, cy2 = centroids[j]
                
                # Calculate Euclidean distance
                distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    def _remove_duplicate_masks(self, masks: List[Dict], min_overlap_percentage: float = 20.0) -> Tuple[List[Dict], List[Dict]]:
        """Remove duplicate masks based on overlap."""
        print(f"Removing duplicate masks (overlap threshold: {min_overlap_percentage}%)...")
        
        if len(masks) <= 1:
            return masks, []
        
        # Sort masks by area (smallest first) to prioritize keeping smaller masks
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=False)
        
        masks_to_keep = []
        removed_masks = []
        
        for i, current_mask in enumerate(sorted_masks):
            is_duplicate = False
            current_bbox = current_mask['bbox']
            
            # Check overlap with all masks we've already decided to keep
            for kept_mask in masks_to_keep:
                kept_bbox = kept_mask['bbox']
                
                # Calculate overlap percentage
                overlap_pct = self._calculate_bbox_overlap_percentage(current_bbox, kept_bbox)
                
                if overlap_pct >= min_overlap_percentage:
                    # Current mask is a duplicate of a smaller mask we're keeping
                    print(f"  Removing mask {current_mask['original_sam_id']} (area: {current_mask['area']:.0f}) - "
                          f"{overlap_pct:.1f}% overlap with mask {kept_mask['original_sam_id']} (area: {kept_mask['area']:.0f})")
                    removed_masks.append(current_mask)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                masks_to_keep.append(current_mask)
        
        print(f"Removed {len(removed_masks)} duplicate masks, {len(masks_to_keep)} remaining")
        return masks_to_keep, removed_masks
    
    def _calculate_bbox_overlap_percentage(self, bbox1: List, bbox2: List) -> float:
        """Calculate percentage of bbox1 covered by bbox2."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate bbox1 area
        bbox1_area = w1 * h1
        if bbox1_area == 0:
            return 0.0
        
        # Calculate intersection coordinates
        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)
        
        # Check if there's actually an intersection
        if left >= right or top >= bottom:
            return 0.0
        
        # Calculate intersection area
        intersection_width = right - left
        intersection_height = bottom - top
        intersection_area = intersection_width * intersection_height
        
        # Calculate percentage of bbox1 covered by bbox2
        overlap_percentage = (intersection_area / bbox1_area) * 100.0
        return overlap_percentage
    
    def _calculate_droplet_statistics(self, final_masks: List[Dict]) -> Dict[str, Any]:
        """Calculate droplet-specific statistics."""
        if not final_masks:
            return {
                'mask_count': 0,
                'average_diameter': 0.0,
                'diameter_std': 0.0,
                'min_diameter': 0.0,
                'max_diameter': 0.0,
                'total_area': 0.0,
                'average_area': 0.0
            }
        
        # Calculate diameters from areas (assuming circular droplets)
        diameters = []
        areas = []
        
        for mask in final_masks:
            area = mask['area']
            # Diameter = 2 * sqrt(area / π)
            diameter = 2 * np.sqrt(area / np.pi)
            diameters.append(diameter)
            areas.append(area)
        
        diameters = np.array(diameters)
        areas = np.array(areas)
        
        statistics = {
            'mask_count': len(final_masks),
            'average_diameter': float(np.mean(diameters)),
            'diameter_std': float(np.std(diameters)),
            'min_diameter': float(np.min(diameters)),
            'max_diameter': float(np.max(diameters)),
            'total_area': float(np.sum(areas)),
            'average_area': float(np.mean(areas)),
            'diameter_distribution': {
                'mean': float(np.mean(diameters)),
                'median': float(np.median(diameters)),
                'std': float(np.std(diameters)),
                'min': float(np.min(diameters)),
                'max': float(np.max(diameters))
            }
        }
        
        return statistics
    
    def create_droplet_visualization(self, image: np.ndarray, final_masks: List[Dict], output_dir: str) -> str:
        """Create visualization showing all droplet masks."""
        print(f"\nCreating droplet visualization...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(image)
        
        # Draw each mask with uniform color
        drawn_masks = 0
        uniform_color = (0.5, 0, 0.5)  # Purple color for all bounding boxes
        
        for i, mask_data in enumerate(final_masks):
            bbox = mask_data['bbox']
            x, y, w, h = bbox
            
            # Convert to integers
            try:
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Skip masks with invalid dimensions
                if w <= 0 or h <= 0:
                    continue
                    
                # Skip masks outside image bounds
                if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                    continue
                    
            except (ValueError, TypeError):
                continue
            
            # Draw bounding box with uniform color
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=uniform_color, facecolor='none')
            ax.add_patch(rect)
            
            # Add mask ID
            ax.text(x, y - 5, f"{i+1}", fontsize=8, color='white', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8),
                   ha='left', va='bottom', weight='bold')
            
            drawn_masks += 1
        
        print(f"Successfully drew {drawn_masks}/{len(final_masks)} masks")
        
        # Create title with statistics
        stats = self._calculate_droplet_statistics(final_masks)
        title = f"Droplet Analysis\n(Total: {stats['mask_count']}, Avg Diameter: {stats['average_diameter']:.1f}px)"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"droplet_analysis_{timestamp}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved droplet visualization: {output_path}")
        return output_path 