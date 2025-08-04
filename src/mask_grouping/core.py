#!/usr/bin/env python3
"""
Core mask grouping processing algorithms.
Contains all the logic from the original mask_size_grouping.py script.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class MaskGroupingProcessor:
    """
    Main processor class that implements all functionality from mask_size_grouping.py.
    
    This class handles:
    - SAM mask generation with exact same parameters
    - Edge proximity filtering
    - Circularity filtering
    - Blob distance filtering
    - K-means clustering
    - Overlap analysis
    - Visualization generation
    """
    
    def __init__(self, mask_generator, overlap_threshold=0.8, min_circularity=0.53, max_blob_distance=50, fluorescent_mode=False, min_brightness_threshold=200):
        """
        Initialize the mask grouping processor.
        
        Args:
            mask_generator: SAM mask generator (configured with same params as original script)
            overlap_threshold: Minimum overlap percentage for analysis (default 80%)
            min_circularity: Minimum circularity threshold (default 0.53)
            max_blob_distance: Maximum blob distance in pixels (default 50)
            fluorescent_mode: Enable fluorescent mode with brightness filtering (default False)
            min_brightness_threshold: Minimum brightness threshold for fluorescent mode (default 200)
        """
        self.mask_generator = mask_generator
        self.overlap_threshold = overlap_threshold
        self.min_circularity = min_circularity
        self.max_blob_distance = max_blob_distance
        self.fluorescent_mode = fluorescent_mode
        self.min_brightness_threshold = min_brightness_threshold
        
    def process_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image array through the complete mask grouping pipeline.
        
        This replicates the exact same workflow as mask_size_grouping.py:
        1. Generate masks using SAM
        2. Add circularity information
        3. Filter edge-touching masks
        4. Filter by circularity
        5. Filter by blob distance
        6. Cluster using K-means
        7. Analyze overlaps
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary containing all results and statistics
        """
        print(f"Processing image: {image.shape}")
        
        # Step 1: Generate masks using SAM (same parameters as original script)
        print("Generating masks...")
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
        
        # Step 6: Cluster masks using K-means
        clustered_masks, cluster_labels = self._cluster_masks_kmeans(filtered_masks, n_clusters=2)
        
        # Step 6.5: Filter smaller masks by brightness (fluorescent mode only)
        removed_low_brightness = []
        if self.fluorescent_mode:
            print(f"\nFLUORESCENT MODE: Applying brightness filter to smaller masks...")
            
            # Determine which cluster has smaller masks
            if len(clustered_masks) >= 2:
                avg_area_0 = np.mean([mask['area'] for mask in clustered_masks[0]]) if clustered_masks[0] else 0
                avg_area_1 = np.mean([mask['area'] for mask in clustered_masks[1]]) if clustered_masks[1] else 0
                
                if avg_area_0 < avg_area_1:
                    smaller_cluster_idx = 0
                    larger_cluster_idx = 1
                else:
                    smaller_cluster_idx = 1
                    larger_cluster_idx = 0
                
                # Apply brightness filter only to smaller masks
                smaller_masks_filtered, removed_low_brightness = self._filter_by_brightness(
                    clustered_masks[smaller_cluster_idx], image, self.min_brightness_threshold
                )
                
                # Update the clustered masks with filtered smaller masks
                clustered_masks[smaller_cluster_idx] = smaller_masks_filtered
                
                print(f"  Applied brightness filter to smaller cluster (cluster {smaller_cluster_idx})")
                print(f"  Removed {len(removed_low_brightness)} low brightness masks from smaller cluster")
            else:
                print(f"  Warning: Not enough clusters for brightness filtering")
        else:
            print(f"\nNON-FLUORESCENT MODE: Skipping brightness filter")
        
        # Step 7: Remove duplicate masks within each cluster (removes larger mask when overlap > threshold) 
        clustered_masks, removed_duplicates_per_cluster = self._remove_duplicate_masks_within_clusters(
            clustered_masks, min_overlap_percentage=20.0
        )
        
        # Step 8: Analyze mask overlaps using statistical aggregate detection
        overlap_results = self._analyze_mask_overlaps(clustered_masks, image)
        
        # Compile statistics
        statistics = self._compile_statistics(
            all_masks, filtered_masks, clustered_masks, 
            removed_edge, removed_circularity, removed_multi_blob,
            removed_low_brightness, removed_duplicates_per_cluster, overlap_results
        )
        
        return {
            'total_masks': len(all_masks),
            'filtered_masks': len(filtered_masks),
            'clustered_masks': clustered_masks,
            'final_masks': overlap_results['final_masks'],
            'overlap_summary': overlap_results['overlap_summary'],
            'smaller_cluster_masks': overlap_results.get('smaller_cluster_masks', []),
            'statistics': statistics,
            'processing_details': {
                'removed_edge': len(removed_edge),
                'removed_circularity': len(removed_circularity),
                'removed_multi_blob': len(removed_multi_blob),
                'removed_low_brightness': len(removed_low_brightness),
                'removed_duplicates_per_cluster': [len(removed) for removed in removed_duplicates_per_cluster],
                'cluster_counts': [len(cluster) for cluster in clustered_masks],
                'fluorescent_mode': self.fluorescent_mode,
                'min_brightness_threshold': self.min_brightness_threshold
            }
        }
    
    def _add_circularity_to_masks(self, masks: List[Dict]) -> List[Dict]:
        """Add circularity information to all masks (same as original script)."""
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
        """Calculate circularity of a mask (4π*area/perimeter²) - same as original."""
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
        """Filter out edge-touching masks (same as original script)."""
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
        """Filter masks by circularity threshold (same as original script)."""
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
        num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)
        
        # If only one blob (plus background), return 0
        if num_labels <= 2:  # background + 1 blob
            return 0.0
        
        # Calculate centroids of each blob
        centroids = []
        for label in range(1, num_labels):  # Skip background (label 0)
            blob_pixels = np.where(labels == label)
            if len(blob_pixels[0]) > 0:
                centroid_y = np.mean(blob_pixels[0])
                centroid_x = np.mean(blob_pixels[1])
                centroids.append((centroid_x, centroid_y))
        
        # Calculate maximum distance between any two centroids
        max_distance = 0.0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distance = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                                 (centroids[i][1] - centroids[j][1])**2)
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    def _cluster_masks_kmeans(self, masks: List[Dict], n_clusters: int = 2) -> Tuple[List[List[Dict]], np.ndarray]:
        """Cluster masks into groups using K-means (same as original script)."""
        print(f"Clustering masks into {n_clusters} groups using K-means...")
        
        # Extract features
        features = self._extract_mask_features(masks)
        
        # Apply log transformation to area to handle extreme values
        features_processed = features.copy()
        features_processed[:, 0] = np.log1p(features_processed[:, 0])  # log(area + 1)
        
        # Normalize features for better clustering
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_processed)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Group masks by cluster
        clustered_masks = [[] for _ in range(n_clusters)]
        for i, label in enumerate(cluster_labels):
            clustered_masks[label].append(masks[i])
        
        # Print cluster statistics
        for i, cluster in enumerate(clustered_masks):
            if len(cluster) > 0:
                areas = [mask['area'] for mask in cluster]
                print(f"Cluster {i}: {len(cluster)} masks, "
                      f"area range: {min(areas):.0f}-{max(areas):.0f}, "
                      f"mean area: {np.mean(areas):.0f}")
            else:
                print(f"Cluster {i}: 0 masks")
        
        return clustered_masks, cluster_labels
    
    def _remove_duplicate_masks_within_clusters(self, clustered_masks: List[List[Dict]], 
                                               min_overlap_percentage: float = 20.0) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Remove duplicate masks within each cluster based on bounding box overlap.
        When two masks overlap more than min_overlap_percentage, the smaller mask is removed.
        
        This implements the exact same functionality as mask_size_grouping.py
        
        Args:
            clustered_masks: List of clusters, each containing list of mask dictionaries
            min_overlap_percentage: Minimum overlap percentage to consider masks as duplicates
            
        Returns:
            tuple: (cleaned_clustered_masks, removed_duplicates_per_cluster)
        """
        print("\n" + "="*60)
        print(f"REMOVING DUPLICATE MASKS WITHIN CLUSTERS (overlap > {min_overlap_percentage}%)")
        print("="*60)
        
        cleaned_clusters = []
        removed_duplicates_per_cluster = []
        total_removed = 0
        
        for cluster_idx, cluster_masks in enumerate(clustered_masks):
            print(f"\nProcessing Cluster {cluster_idx}: {len(cluster_masks)} masks")
            
            if len(cluster_masks) <= 1:
                # No duplicates possible with 0 or 1 mask
                cleaned_clusters.append(cluster_masks.copy())
                removed_duplicates_per_cluster.append([])
                continue
            
            # Sort masks by area (largest first) to prioritize keeping larger masks
            sorted_masks = sorted(cluster_masks, key=lambda x: x['area'], reverse=True)
            
            masks_to_keep = []
            removed_masks = []
            
            for i, current_mask in enumerate(sorted_masks):
                is_duplicate = False
                current_bbox = current_mask['bbox']
                
                # Check overlap with all masks we've already decided to keep
                for kept_mask in masks_to_keep:
                    kept_bbox = kept_mask['bbox']
                    
                    # Calculate overlap percentage (what % of current mask overlaps with kept mask)
                    overlap_pct = self._calculate_bbox_overlap_percentage(current_bbox, kept_bbox)
                    
                    if overlap_pct >= min_overlap_percentage:
                        # Current mask is a duplicate of a larger mask we're keeping
                        print(f"  Removing mask {current_mask['original_sam_id']} (area: {current_mask['area']:.0f}) - "
                              f"{overlap_pct:.1f}% overlap with mask {kept_mask['original_sam_id']} (area: {kept_mask['area']:.0f})")
                        removed_masks.append(current_mask)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    masks_to_keep.append(current_mask)
            
            print(f"  Cluster {cluster_idx} results: kept {len(masks_to_keep)}, removed {len(removed_masks)} duplicates")
            
            cleaned_clusters.append(masks_to_keep)
            removed_duplicates_per_cluster.append(removed_masks)
            total_removed += len(removed_masks)
        
        print(f"\nDUPLICATE REMOVAL SUMMARY:")
        print(f"Total masks removed across all clusters: {total_removed}")
        for i, (original_count, cleaned_count, removed_count) in enumerate(
            zip([len(cluster) for cluster in clustered_masks],
                [len(cluster) for cluster in cleaned_clusters], 
                [len(removed) for removed in removed_duplicates_per_cluster])):
            print(f"  Cluster {i}: {original_count} → {cleaned_count} (removed {removed_count})")
        
        return cleaned_clusters, removed_duplicates_per_cluster
    
    def _cluster_smaller_masks_kmeans(self, smaller_masks: List[Dict], n_clusters: int = 2) -> Tuple[List[List[Dict]], np.ndarray]:
        """Perform second-level K-means clustering on smaller masks for aggregation detection."""
        print(f"\n" + "="*60)
        print(f"SECOND-LEVEL CLUSTERING: Clustering smaller masks into {n_clusters} groups")
        print("="*60)
        
        if len(smaller_masks) < n_clusters:
            print(f"Not enough smaller masks ({len(smaller_masks)}) to cluster into {n_clusters} groups")
            return [smaller_masks] + [[] for _ in range(n_clusters - 1)], np.zeros(len(smaller_masks))
        
        # Extract features for smaller masks
        features = self._extract_mask_features(smaller_masks)
        
        # Apply log transformation to area to handle extreme values
        features_processed = features.copy()
        features_processed[:, 0] = np.log1p(features_processed[:, 0])  # log(area + 1)
        
        # Normalize features for better clustering
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_processed)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Check clustering balance
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"Second-level clustering balance: {counts}")
        
        # Group masks by cluster
        clustered_smaller_masks = [[] for _ in range(n_clusters)]
        for i, label in enumerate(cluster_labels):
            clustered_smaller_masks[label].append(smaller_masks[i])
        
        # Print cluster statistics
        for i, cluster in enumerate(clustered_smaller_masks):
            if len(cluster) > 0:
                areas = [mask['area'] for mask in cluster]
                print(f"Small Cluster {i}: {len(cluster)} masks, "
                      f"area range: {min(areas):.0f}-{max(areas):.0f}, "
                      f"mean area: {np.mean(areas):.0f}")
            else:
                print(f"Small Cluster {i}: 0 masks")
        
        print(f"Second-level clustering completed: {len(smaller_masks)} → {[len(c) for c in clustered_smaller_masks]}")
        
        return clustered_smaller_masks, cluster_labels
    
    def _create_overlap_count_statistics(self, final_masks: List[Dict], aggregate_masks: List[Dict], simple_masks: List[Dict]) -> Dict[str, Any]:
        """
        Create detailed statistical analysis of overlap counts.
        
        For aggregation masks, classify them as 'aggregation group' regardless of overlap count.
        For simple masks, show distribution by actual overlap count.
        """
        print(f"\n" + "="*60)
        print("CREATING OVERLAP COUNT STATISTICS")
        print("="*60)
        
        # Initialize statistics structure
        statistics = {
            'overlap_count_distribution': {},
            'aggregation_group_analysis': {},
            'simple_group_analysis': {},
            'summary': {}
        }
        
        # Analyze simple masks (non-aggregation) by overlap count
        simple_overlap_counts = {}
        for mask in simple_masks:
            overlap_count = mask['overlap_count']
            if overlap_count not in simple_overlap_counts:
                simple_overlap_counts[overlap_count] = []
            simple_overlap_counts[overlap_count].append(mask)
        
        # Analyze aggregate masks (classify as aggregation group)
        aggregate_overlap_counts = {}
        for mask in aggregate_masks:
            overlap_count = mask['overlap_count']
            if overlap_count not in aggregate_overlap_counts:
                aggregate_overlap_counts[overlap_count] = []
            aggregate_overlap_counts[overlap_count].append(mask)
        
        # Create overlap count distribution (combining simple and aggregate)
        all_overlap_counts = {}
        for mask in final_masks:
            overlap_count = mask['overlap_count']
            is_aggregate = mask.get('is_aggregate_overlap', False)
            
            if overlap_count not in all_overlap_counts:
                all_overlap_counts[overlap_count] = {
                    'total_masks': 0,
                    'simple_masks': 0,
                    'aggregate_masks': 0,
                    'mask_ids': []
                }
            
            all_overlap_counts[overlap_count]['total_masks'] += 1
            all_overlap_counts[overlap_count]['mask_ids'].append(mask['original_sam_id'])
            
            if is_aggregate:
                all_overlap_counts[overlap_count]['aggregate_masks'] += 1
            else:
                all_overlap_counts[overlap_count]['simple_masks'] += 1
        
        # Sort by overlap count for better readability
        sorted_overlap_counts = dict(sorted(all_overlap_counts.items()))
        statistics['overlap_count_distribution'] = sorted_overlap_counts
        
        # Aggregation group analysis (all aggregate masks regardless of count)
        statistics['aggregation_group_analysis'] = {
            'total_aggregation_masks': len(aggregate_masks),
            'aggregation_overlap_breakdown': {
                str(k): {
                    'count': len(v),
                    'mask_ids': [mask['original_sam_id'] for mask in v],
                    'percentage': (len(v) / len(aggregate_masks) * 100) if len(aggregate_masks) > 0 else 0
                }
                for k, v in sorted(aggregate_overlap_counts.items())
            },
            'total_overlaps_in_aggregation': sum([mask['overlap_count'] for mask in aggregate_masks]),
            'average_overlaps_per_aggregation_mask': np.mean([mask['overlap_count'] for mask in aggregate_masks]) if aggregate_masks else 0
        }
        
        # Simple group analysis (by actual overlap count)
        statistics['simple_group_analysis'] = {
            'total_simple_masks': len(simple_masks),
            'simple_overlap_breakdown': {
                str(k): {
                    'count': len(v),
                    'mask_ids': [mask['original_sam_id'] for mask in v],
                    'percentage': (len(v) / len(simple_masks) * 100) if len(simple_masks) > 0 else 0
                }
                for k, v in sorted(simple_overlap_counts.items())
            },
            'total_overlaps_in_simple': sum([mask['overlap_count'] for mask in simple_masks]),
            'average_overlaps_per_simple_mask': np.mean([mask['overlap_count'] for mask in simple_masks]) if simple_masks else 0
        }
        
        # Overall summary
        statistics['summary'] = {
            'total_masks_analyzed': len(final_masks),
            'masks_with_no_overlaps': all_overlap_counts.get(0, {}).get('total_masks', 0),
            'masks_with_overlaps': len(final_masks) - all_overlap_counts.get(0, {}).get('total_masks', 0),
            'aggregation_percentage': (len(aggregate_masks) / len(final_masks) * 100) if final_masks else 0,
            'simple_percentage': (len(simple_masks) / len(final_masks) * 100) if final_masks else 0,
            'max_overlaps_found': max([mask['overlap_count'] for mask in final_masks]) if final_masks else 0,
            'min_overlaps_found': min([mask['overlap_count'] for mask in final_masks]) if final_masks else 0
        }
        
        # Print detailed statistics
        print(f"OVERLAP COUNT DISTRIBUTION:")
        for overlap_count, data in sorted_overlap_counts.items():
            total = data['total_masks']
            simple = data['simple_masks']
            aggregate = data['aggregate_masks']
            print(f"  {overlap_count} overlaps: {total} masks (Simple: {simple}, Aggregate: {aggregate})")
        
        print(f"\nAGGREGATION GROUP ANALYSIS:")
        print(f"  Total aggregation masks: {len(aggregate_masks)}")
        print(f"  Average overlaps per aggregation mask: {statistics['aggregation_group_analysis']['average_overlaps_per_aggregation_mask']:.1f}")
        
        print(f"\nSIMPLE GROUP ANALYSIS:")
        print(f"  Total simple masks: {len(simple_masks)}")
        print(f"  Average overlaps per simple mask: {statistics['simple_group_analysis']['average_overlaps_per_simple_mask']:.1f}")
        
        print(f"\nSUMMARY:")
        print(f"  Masks with no overlaps: {statistics['summary']['masks_with_no_overlaps']}")
        print(f"  Masks with overlaps: {statistics['summary']['masks_with_overlaps']}")
        print(f"  Aggregation percentage: {statistics['summary']['aggregation_percentage']:.1f}%")
        print(f"  Simple percentage: {statistics['summary']['simple_percentage']:.1f}%")
        
        return statistics
    
    def _extract_mask_features(self, masks: List[Dict]) -> np.ndarray:
        """Extract features for K-means clustering (same as original script)."""
        features = []
        
        for mask_data in masks:
            bbox = mask_data['bbox']
            
            # Feature vector: [area, bbox_width, bbox_height, aspect_ratio, stability_score, circularity]
            area = mask_data['area']
            width = bbox[2]
            height = bbox[3]
            aspect_ratio = width / height if height > 0 else 1.0
            stability_score = mask_data['stability_score']
            circularity = mask_data.get('circularity', 0.0)
            
            features.append([
                area,
                width,
                height,
                aspect_ratio,
                stability_score,
                circularity
            ])
        
        return np.array(features)
    
    def _analyze_mask_overlaps(self, clustered_masks: List[List[Dict]], image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze overlaps between smaller and larger masks with statistical aggregate detection.
        
        This implements statistical threshold-based aggregate group detection:
        - Statistical analysis of smaller masks (area and circularity)
        - Threshold-based aggregate group identification
        - Enhanced overlap classification (Simple vs Aggregate)
        """
        print("\n" + "="*60)
        print("OVERLAP ANALYSIS: Finding smaller masks that overlap with larger masks")
        print("="*60)
        
        if len(clustered_masks) < 2:
            print("Not enough clusters for overlap analysis")
            return {'final_masks': [], 'overlap_summary': {}}
        
        cluster_0 = clustered_masks[0]
        cluster_1 = clustered_masks[1]
        
        # Determine which cluster has larger masks on average
        if cluster_0 and cluster_1:
            avg_area_0 = np.mean([mask['area'] for mask in cluster_0])
            avg_area_1 = np.mean([mask['area'] for mask in cluster_1])
            
            if avg_area_0 > avg_area_1:
                larger_masks = cluster_0
                smaller_masks = cluster_1
                print(f"Cluster 0 identified as LARGER masks (avg area: {avg_area_0:.0f})")
                print(f"Cluster 1 identified as SMALLER masks (avg area: {avg_area_1:.0f})")
            else:
                larger_masks = cluster_1
                smaller_masks = cluster_0
                print(f"Cluster 1 identified as LARGER masks (avg area: {avg_area_1:.0f})")
                print(f"Cluster 0 identified as SMALLER masks (avg area: {avg_area_0:.0f})")
        else:
            print("One cluster is empty, using available cluster as larger masks")
            larger_masks = cluster_0 if cluster_0 else cluster_1
            smaller_masks = cluster_1 if cluster_0 else cluster_0
        
        # NEW: Use statistical threshold-based aggregate detection
        aggregate_smaller_masks, simple_smaller_masks, aggregate_analysis = self._identify_aggregate_groups_statistical(
            smaller_masks, area_threshold=0.90, circularity_threshold=None
        )
        
        # Analyze each larger mask for overlaps with smaller masks
        final_masks = []
        overlap_details = []
        min_overlap_percentage = self.overlap_threshold * 100  # Convert to percentage
        
        print(f"\nAnalyzing overlaps with minimum {min_overlap_percentage}% coverage...")
        print(f"Large masks to analyze: {len(larger_masks)}")
        print(f"Small masks to check for overlap: {len(smaller_masks)}")
        print(f"  - Aggregate smaller masks: {len(aggregate_smaller_masks)}")
        print(f"  - Simple smaller masks: {len(simple_smaller_masks)}")
        
        for i, large_mask in enumerate(larger_masks):
            large_bbox = large_mask['bbox']
            overlapping_small_masks = []
            overlap_percentages = []
            overlapping_aggregate_masks = []  # Masks from the aggregate group
            overlapping_simple_masks = []     # Masks from the simple group
            
            # Check overlap with each smaller mask
            for j, small_mask in enumerate(smaller_masks):
                small_bbox = small_mask['bbox']
                
                # Calculate overlap percentage using bounding boxes
                overlap_pct = self._calculate_bbox_overlap_percentage(small_bbox, large_bbox)
                
                if overlap_pct >= min_overlap_percentage:
                    overlap_info = {
                        'small_mask_id': small_mask['original_sam_id'],
                        'small_mask_index': j,
                        'overlap_percentage': overlap_pct,
                        'small_mask_area': small_mask['area'],
                        'small_mask_circularity': small_mask.get('circularity', 0.0)
                    }
                    
                    overlapping_small_masks.append(overlap_info)
                    overlap_percentages.append(overlap_pct)
                    
                    # Separate into aggregate vs simple groups based on statistical analysis
                    if small_mask['original_sam_id'] in [mask['original_sam_id'] for mask in aggregate_smaller_masks]:
                        overlapping_aggregate_masks.append(overlap_info)
                    else:
                        overlapping_simple_masks.append(overlap_info)
            
            # Statistical overlap classification logic
            num_aggregate_overlaps = len(overlapping_aggregate_masks)
            num_simple_overlaps = len(overlapping_simple_masks)
            total_overlaps = len(overlapping_small_masks)
            
            # Determine overlap type based on statistical logic
            if num_aggregate_overlaps > 0:
                # Contains aggregate masks = Aggregate group
                is_aggregate = True
                overlap_type = 'aggregate'
                overlap_label = f"AGG({total_overlaps})"
            else:
                # Contains only simple masks = Simple count
                is_aggregate = False
                overlap_type = 'simple'
                overlap_label = f"{total_overlaps}"
            
            # Add enhanced overlap information to the large mask
            final_mask = large_mask.copy()
            final_mask['overlap_count'] = total_overlaps
            final_mask['overlapping_masks'] = overlapping_small_masks
            final_mask['overlapping_aggregate_masks'] = overlapping_aggregate_masks
            final_mask['overlapping_simple_masks'] = overlapping_simple_masks
            final_mask['num_aggregate_overlaps'] = num_aggregate_overlaps
            final_mask['num_simple_overlaps'] = num_simple_overlaps
            final_mask['is_aggregate_overlap'] = is_aggregate
            final_mask['overlap_type'] = overlap_type
            final_mask['overlap_label'] = overlap_label
            final_mask['avg_overlap_percentage'] = np.mean(overlap_percentages) if overlap_percentages else 0.0
            
            final_masks.append(final_mask)
            
            # Store detailed information with statistical aggregation data
            overlap_detail = {
                'large_mask_id': large_mask['original_sam_id'],
                'large_mask_area': large_mask['area'],
                'overlap_count': total_overlaps,
                'num_aggregate_overlaps': num_aggregate_overlaps,
                'num_simple_overlaps': num_simple_overlaps,
                'overlapping_small_masks': overlapping_small_masks,
                'overlapping_aggregate_masks': overlapping_aggregate_masks,
                'overlapping_simple_masks': overlapping_simple_masks,
                'is_aggregate_overlap': is_aggregate,
                'overlap_type': overlap_type,
                'avg_overlap_percentage': np.mean(overlap_percentages) if overlap_percentages else 0.0
            }
            overlap_details.append(overlap_detail)
            
            if i < 10:  # Print details for first 10 masks
                overlap_type_str = "AGGREGATE" if is_aggregate else "SIMPLE"
                print(f"Large mask {large_mask['original_sam_id']}: {total_overlaps} overlapping masks ({overlap_type_str})")
                print(f"  - Aggregate overlaps: {num_aggregate_overlaps}")
                print(f"  - Simple overlaps: {num_simple_overlaps}")
                if overlapping_small_masks:
                    for overlap in overlapping_small_masks:
                        group_type = "AGGREGATE" if overlap in overlapping_aggregate_masks else "SIMPLE"
                        print(f"  - Small mask {overlap['small_mask_id']} ({group_type}): {overlap['overlap_percentage']:.1f}% overlap")
        
        # Create summary statistics
        overlap_counts = [mask['overlap_count'] for mask in final_masks]
        if len(overlap_counts) > 0:
            unique_counts, count_frequencies = np.unique(overlap_counts, return_counts=True)
        else:
            unique_counts, count_frequencies = np.array([]), np.array([])
        
        # Separate aggregate and simple overlaps based on statistical logic
        aggregate_masks = [mask for mask in final_masks if mask['is_aggregate_overlap']]
        simple_masks = [mask for mask in final_masks if not mask['is_aggregate_overlap']]
        
        print(f"\nSTATISTICAL OVERLAP SUMMARY:")
        print(f"Total large masks analyzed: {len(final_masks)}")
        print(f"  Simple overlaps (only simple group): {len(simple_masks)} masks")
        print(f"  Aggregate overlaps (contains aggregate group): {len(aggregate_masks)} masks")
        for count, freq in zip(unique_counts, count_frequencies):
            print(f"  {freq} large masks have {count} total overlapping masks")
        
        # Show breakdown by aggregate vs simple group overlaps
        if final_masks:
            aggregate_group_overlaps = [mask['num_aggregate_overlaps'] for mask in final_masks]
            simple_group_overlaps = [mask['num_simple_overlaps'] for mask in final_masks]
            print(f"\nDETAILED BREAKDOWN:")
            print(f"  Total aggregate group overlaps: {sum(aggregate_group_overlaps)}")
            print(f"  Total simple group overlaps: {sum(simple_group_overlaps)}")
        
        if aggregate_masks:
            print(f"\nAGGREGATE GROUP DETAILS:")
            for mask in aggregate_masks[:5]:  # Show first 5 aggregate overlaps
                print(f"  Mask {mask['original_sam_id']}: {mask['num_simple_overlaps']} simple + {mask['num_aggregate_overlaps']} aggregate = {mask['overlap_count']} total")
        
        # Create detailed overlap count statistics
        overlap_count_statistics = self._create_overlap_count_statistics(final_masks, aggregate_masks, simple_masks)
        
        overlap_summary = {
            'total_large_masks': len(final_masks),
            'simple_overlaps': len(simple_masks),
            'aggregate_overlaps': len(aggregate_masks),
            'overlap_distribution': dict(zip([int(c) for c in unique_counts], [int(f) for f in count_frequencies])),
            'overlap_count_statistics': overlap_count_statistics,
            'min_overlap_percentage': min_overlap_percentage,
            'overlap_details': overlap_details,
            'total_overlaps': int(sum(overlap_counts)) if overlap_counts else 0,
            'statistical_aggregate_analysis': aggregate_analysis,
            'statistical_logic': {
                'total_aggregate_group_overlaps': sum([mask['num_aggregate_overlaps'] for mask in final_masks]),
                'total_simple_group_overlaps': sum([mask['num_simple_overlaps'] for mask in final_masks])
            }
        }
        
        return {
            'final_masks': final_masks,
            'overlap_summary': overlap_summary,
            'larger_cluster_masks': larger_masks,
            'smaller_cluster_masks': smaller_masks,
            'aggregate_smaller_masks': aggregate_smaller_masks,
            'simple_smaller_masks': simple_smaller_masks,
            'aggregate_masks': aggregate_masks,
            'simple_masks': simple_masks,
            'statistical_aggregate_analysis': aggregate_analysis
        }
    
    def _calculate_bbox_overlap_percentage(self, bbox1: List, bbox2: List) -> float:
        """Calculate the percentage of bbox1 that overlaps with bbox2 using bounding box coordinates."""
        # bbox format: [x, y, width, height]
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
    
    def create_overlap_visualization(self, image: np.ndarray, final_masks: List[Dict], output_dir: str, smaller_masks: List[Dict] = None) -> str:
        """Create visualization showing larger masks labeled with overlap counts and only overlapping smaller masks."""
        print(f"\nCreating overlap visualization...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(image)
        
        # Collect all overlapping smaller masks from final_masks
        overlapping_small_masks = set()
        for mask_data in final_masks:
            if 'overlapping_masks' in mask_data:
                for overlap_info in mask_data['overlapping_masks']:
                    overlapping_small_masks.add(overlap_info['small_mask_id'])
        
        # Draw only overlapping smaller masks bounding boxes
        if smaller_masks and overlapping_small_masks:
            print(f"Drawing {len(overlapping_small_masks)} overlapping smaller masks...")
            drawn_smaller = 0
            for mask_data in smaller_masks:
                # Only draw if this mask is in the overlapping set
                if mask_data['original_sam_id'] in overlapping_small_masks:
                    bbox = mask_data['bbox']
                    x, y, w, h = bbox
                    
                    # Convert to integers and validate
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
                    
                    # Draw smaller mask bounding box in cyan with thin border
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                           edgecolor='cyan', facecolor='none', alpha=0.6)
                    ax.add_patch(rect)
                    drawn_smaller += 1
            
            print(f"Successfully drew {drawn_smaller}/{len(overlapping_small_masks)} overlapping smaller masks")
        else:
            print("No overlapping smaller masks to display")
        
        # Create color map for different overlap counts
        max_overlap = max([mask['overlap_count'] for mask in final_masks]) if final_masks else 0
        colors = plt.cm.viridis(np.linspace(0, 1, max_overlap + 1))
        
        # Draw each final mask with enhanced aggregation detection visualization
        drawn_masks = 0
        for mask_data in final_masks:
            bbox = mask_data['bbox']
            overlap_count = mask_data['overlap_count']
            is_aggregate = mask_data.get('is_aggregate_overlap', False)
            overlap_label = mask_data.get('overlap_label', str(overlap_count))
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
            
            # Enhanced color scheme based on aggregation detection
            if is_aggregate:
                # Use red colors for aggregate overlaps
                color = plt.cm.Reds(0.5 + 0.5 * (overlap_count / max(max_overlap, 1)))
                linewidth = 3  # Thicker border for aggregate
                bg_color = 'red'
            else:
                # Use original viridis colors for simple overlaps
                color = colors[overlap_count] if overlap_count <= max_overlap else colors[-1]
                linewidth = 2
                bg_color = 'black'
            
            # Draw bounding box with enhanced styling
            rect = patches.Rectangle((x, y), w, h, linewidth=linewidth, 
                                   edgecolor=color[:3], facecolor='none')
            ax.add_patch(rect)
            
            # Add overlap label without ID number
            label_text = f"{overlap_label}"
            ax.text(x, y - 5, label_text, fontsize=6, color='white', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color, alpha=0.8),
                   ha='left', va='bottom', weight='bold')
            
            drawn_masks += 1
        
        print(f"Successfully drew {drawn_masks}/{len(final_masks)} masks")
        
        # Count aggregate vs simple overlaps for title
        aggregate_count = len([m for m in final_masks if m.get('is_aggregate_overlap', False)])
        simple_count = len(final_masks) - aggregate_count
        
        # Create enhanced title with aggregation information
        title = f"Large Masks with Aggregation Detection\n(Min {self.overlap_threshold*100}% overlap, Simple: {simple_count}, Aggregate: {aggregate_count})\nAGG(n) = Aggregate overlap contains larger group masks"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"overlap_analysis_masks_{self.overlap_threshold*100:.0f}pct_{timestamp}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved overlap visualization: {output_path}")
        return output_path
    
    def _compile_statistics(self, all_masks: List[Dict], filtered_masks: List[Dict], 
                          clustered_masks: List[List[Dict]], removed_edge: List[Dict], 
                          removed_circularity: List[Dict], removed_multi_blob: List[Dict],
                          removed_low_brightness: List[Dict], removed_duplicates_per_cluster: List[List[Dict]], 
                          overlap_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive statistics about the processing pipeline."""
        
        # Calculate basic statistics
        all_areas = [mask['area'] for mask in all_masks]
        all_circularities = [mask.get('circularity', 0.0) for mask in all_masks]
        
        total_removed_duplicates = sum(len(removed) for removed in removed_duplicates_per_cluster)
        
        statistics = {
            'processing_summary': {
                'original_masks': len(all_masks),
                'after_edge_filtering': len(all_masks) - len(removed_edge),
                'after_circularity_filtering': len(all_masks) - len(removed_edge) - len(removed_circularity),
                'after_blob_filtering': len(all_masks) - len(removed_edge) - len(removed_circularity) - len(removed_multi_blob),
                'after_clustering': sum(len(cluster) for cluster in clustered_masks) + len(removed_low_brightness),
                'after_brightness_filtering': sum(len(cluster) for cluster in clustered_masks) if self.fluorescent_mode else sum(len(cluster) for cluster in clustered_masks),
                'after_duplicate_removal': sum(len(cluster) for cluster in clustered_masks) - total_removed_duplicates,
                'final_clustered_masks': sum(len(cluster) for cluster in clustered_masks),
                'removed_duplicates_total': total_removed_duplicates
            },
            'filtering_criteria': {
                'edge_threshold': 5,
                'min_circularity': self.min_circularity,
                'max_blob_distance': self.max_blob_distance,
                'fluorescent_mode': self.fluorescent_mode,
                'min_brightness_threshold': self.min_brightness_threshold if self.fluorescent_mode else None,
                'duplicate_removal_overlap_threshold': 20.0,
                'overlap_analysis_threshold_percent': self.overlap_threshold * 100
            },
            'clustering_info': {
                'method': 'K-means',
                'n_clusters': len(clustered_masks),
                'cluster_sizes': [len(cluster) for cluster in clustered_masks],
                'duplicate_removal_enabled': True,
                'removed_duplicates_per_cluster': [len(removed) for removed in removed_duplicates_per_cluster]
            },
            'overall_statistics': {
                'area': {
                    'min': float(min(all_areas)) if all_areas else 0,
                    'max': float(max(all_areas)) if all_areas else 0,
                    'mean': float(np.mean(all_areas)) if all_areas else 0,
                    'median': float(np.median(all_areas)) if all_areas else 0
                },
                'circularity': {
                    'min': float(min(all_circularities)) if all_circularities else 0,
                    'max': float(max(all_circularities)) if all_circularities else 0,
                    'mean': float(np.mean(all_circularities)) if all_circularities else 0,
                    'median': float(np.median(all_circularities)) if all_circularities else 0
                }
            },
            'overlap_analysis': overlap_results['overlap_summary'] if overlap_results else {}
        }
        
        return statistics
    
    def _identify_aggregate_groups_statistical(self, smaller_masks: List[Dict], area_threshold: float = 0.3, circularity_threshold: float = 0.1) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
        """
        Identify aggregate groups using statistical thresholds based on area and circularity.
        
        Args:
            smaller_masks: List of smaller masks to analyze
            area_threshold: Percentage above mean area to consider as potential aggregate (default 30%)
            circularity_threshold: Minimum circularity difference below mean to consider as aggregate (default 0.1)
        
        Returns:
            Tuple of (aggregate_masks, simple_masks, analysis_info)
        """
        print(f"\n" + "="*60)
        print("STATISTICAL AGGREGATE GROUP IDENTIFICATION")
        print("="*60)
        
        if len(smaller_masks) < 3:
            print(f"Not enough smaller masks ({len(smaller_masks)}) for statistical analysis")
            return [], smaller_masks, {'method': 'insufficient_data', 'thresholds': {}}
        
        # Calculate mean area and circularity
        areas = [mask['area'] for mask in smaller_masks]
        circularities = [mask.get('circularity', 0.0) for mask in smaller_masks]
        
        mean_area = np.mean(areas)
        mean_circularity = np.mean(circularities)
        
        print(f"STATISTICAL THRESHOLDS:")
        print(f"  Mean area: {mean_area:.0f}")
        print(f"  Mean circularity: {mean_circularity:.3f}")
        print(f"  Area threshold: {mean_area * (1 + area_threshold):.0f} (mean + {area_threshold*100:.0f}%)")
        if circularity_threshold is None:
            print(f"  Circularity filter: DISABLED (area only)")
        elif circularity_threshold > 0:
            print(f"  Circularity threshold: {mean_circularity - circularity_threshold:.3f} (mean - {circularity_threshold:.3f})")
        else:
            print(f"  Circularity threshold: {mean_circularity:.3f} (any circularity below mean)")
        
        # Identify potential aggregate masks
        aggregate_masks = []
        simple_masks = []
        
        for mask in smaller_masks:
            area = mask['area']
            circularity = mask.get('circularity', 0.0)
            
            # Check if mask meets aggregate criteria
            area_above_threshold = area >= mean_area * (1 + area_threshold)
            if circularity_threshold is None:
                # Only area criterion, no circularity filter
                is_aggregate = area_above_threshold
            else:
                # Both area and circularity criteria
                if circularity_threshold > 0:
                    circularity_below_threshold = circularity <= (mean_circularity - circularity_threshold)
                else:
                    circularity_below_threshold = circularity < mean_circularity
                is_aggregate = area_above_threshold and circularity_below_threshold
            
            if is_aggregate:
                aggregate_masks.append(mask)
                print(f"  Mask {mask['original_sam_id']}: AGGREGATE (area: {area:.0f}, circularity: {circularity:.3f})")
            else:
                simple_masks.append(mask)
        
        # Calculate statistics
        analysis_info = {
            'method': 'statistical_thresholds',
            'thresholds': {
                'area_threshold_percentage': area_threshold * 100,
                'circularity_threshold_difference': circularity_threshold,
                'mean_area': float(mean_area),
                'mean_circularity': float(mean_circularity),
                'area_threshold_value': float(mean_area * (1 + area_threshold)),
                'circularity_filter_enabled': circularity_threshold is not None,
                'circularity_threshold_value': float(mean_circularity - circularity_threshold) if circularity_threshold and circularity_threshold > 0 else float(mean_circularity)
            },
            'results': {
                'total_smaller_masks': len(smaller_masks),
                'aggregate_masks_count': len(aggregate_masks),
                'simple_masks_count': len(simple_masks),
                'aggregate_percentage': (len(aggregate_masks) / len(smaller_masks) * 100) if smaller_masks else 0
            },
            'aggregate_mask_details': [
                {
                    'mask_id': mask['original_sam_id'],
                    'area': mask['area'],
                    'circularity': mask.get('circularity', 0.0),
                    'area_ratio_to_mean': mask['area'] / mean_area,
                    'circularity_difference_from_mean': mean_circularity - mask.get('circularity', 0.0)
                }
                for mask in aggregate_masks
            ]
        }
        
        print(f"\nAGGREGATE GROUP RESULTS:")
        print(f"  Total smaller masks: {len(smaller_masks)}")
        print(f"  Aggregate masks: {len(aggregate_masks)} ({analysis_info['results']['aggregate_percentage']:.1f}%)")
        print(f"  Simple masks: {len(simple_masks)}")
        
        if aggregate_masks:
            print(f"\nAGGREGATE MASK DETAILS:")
            for detail in analysis_info['aggregate_mask_details']:
                print(f"  Mask {detail['mask_id']}: area={detail['area']:.0f} ({detail['area_ratio_to_mean']:.2f}x mean), "
                      f"circularity={detail['circularity']:.3f} ({detail['circularity_difference_from_mean']:.3f} below mean)")
        
        return aggregate_masks, simple_masks, analysis_info
    
    def _calculate_mask_brightness(self, mask: np.ndarray, image: np.ndarray) -> float:
        """
        Calculate the average brightness within a mask region using HSV Value channel.
        
        Args:
            mask: Binary mask array (True/False or 0/1)
            image: Original image array (RGB or grayscale)
            
        Returns:
            Average brightness within the mask region (0-255)
        """
        # Ensure mask is boolean
        mask_bool = mask.astype(bool)
        
        # Convert image to HSV if it's RGB to get brightness (Value channel)
        if len(image.shape) == 3:
            # Convert RGB to HSV (OpenCV uses BGR by default, but we assume RGB input)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Extract Value channel (brightness)
            brightness_channel = hsv_image[:, :, 2]
        else:
            # If already grayscale, use as brightness
            brightness_channel = image
        
        # Get pixel values within the mask
        masked_pixels = brightness_channel[mask_bool]
        
        if len(masked_pixels) == 0:
            return 0.0
        
        # Calculate average brightness
        avg_brightness = np.mean(masked_pixels)
        return float(avg_brightness)
    
    def _filter_by_brightness(self, masks: List[Dict], image: np.ndarray, min_brightness: float) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter masks by brightness threshold (for fluorescent mode).
        
        Args:
            masks: List of mask dictionaries
            image: Original image array
            min_brightness: Minimum average brightness threshold (0-255)
            
        Returns:
            Tuple of (filtered_masks, removed_low_brightness)
        """
        print(f"Filtering masks by brightness (min_brightness: {min_brightness})...")
        
        filtered_masks = []
        removed_low_brightness = []
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            brightness = self._calculate_mask_brightness(mask, image)
            
            # Store brightness value for later use
            mask_data['pixel_brightness'] = brightness
            
            if brightness >= min_brightness:
                filtered_masks.append(mask_data)
            else:
                removed_low_brightness.append(mask_data)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(masks)} masks...")
        
        print(f"Removed {len(removed_low_brightness)} low brightness masks, {len(filtered_masks)} remaining")
        return filtered_masks, removed_low_brightness

class HugeDropletProcessor:
    """
    Enhanced droplet processor that uses crop_n_layers=3 for more detailed mask generation.
    
    This processor applies the same droplet filtering logic as DropletProcessor but uses
    the more detailed SAM configuration (crop_n_layers=3) for better mask quality.
    """
    
    def __init__(self, mask_generator, min_circularity=0.53, max_blob_distance=50):
        """
        Initialize huge droplet processor.
        
        Args:
            mask_generator: SAM mask generator with crop_n_layers=3
            min_circularity: Minimum circularity threshold
            max_blob_distance: Maximum blob distance threshold
        """
        self.mask_generator = mask_generator
        self.min_circularity = min_circularity
        self.max_blob_distance = max_blob_distance
    
    def process_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image array through enhanced SAM masking and filtering.
        
        This uses the same filtering pipeline as droplet mode but with crop_n_layers=3
        for more detailed and accurate mask generation.
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary containing results and statistics
        """
        print(f"Processing image in huge droplet mode: {image.shape}")
        
        # Step 1: Generate masks using SAM with crop_n_layers=3 (enhanced detail)
        print("Generating masks with enhanced SAM (crop_n_layers=3)...")
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
                'removed_duplicates': len(removed_duplicates),
                'mode': 'huge_droplet',
                'crop_n_layers': 3,
                'points_per_side': 32
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
        num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)
        
        # If only one blob (plus background), return 0
        if num_labels <= 2:  # background + 1 blob
            return 0.0
        
        # Calculate centroids of each blob
        centroids = []
        for label in range(1, num_labels):  # Skip background (label 0)
            blob_pixels = np.where(labels == label)
            if len(blob_pixels[0]) > 0:
                centroid_y = np.mean(blob_pixels[0])
                centroid_x = np.mean(blob_pixels[1])
                centroids.append((centroid_x, centroid_y))
        
        # Calculate maximum distance between any two centroids
        max_distance = 0.0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distance = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                                 (centroids[i][1] - centroids[j][1])**2)
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    def _remove_duplicate_masks(self, masks: List[Dict], min_overlap_percentage: float = 20.0) -> Tuple[List[Dict], List[Dict]]:
        """
        Remove duplicate masks based on bounding box overlap.
        When two masks overlap more than min_overlap_percentage, the smaller mask is removed.
        
        Args:
            masks: List of mask dictionaries
            min_overlap_percentage: Minimum overlap percentage to consider masks as duplicates
            
        Returns:
            tuple: (cleaned_masks, removed_duplicates)
        """
        print(f"\nRemoving duplicate masks (overlap > {min_overlap_percentage}%)...")
        
        if len(masks) <= 1:
            return masks, []
        
        # Sort masks by area (largest first) to prioritize keeping larger masks
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        masks_to_keep = []
        removed_masks = []
        
        for i, current_mask in enumerate(sorted_masks):
            is_duplicate = False
            current_bbox = current_mask['bbox']
            
            # Check overlap with all masks we've already decided to keep
            for kept_mask in masks_to_keep:
                kept_bbox = kept_mask['bbox']
                
                # Calculate overlap percentage (what % of current mask overlaps with kept mask)
                overlap_pct = self._calculate_bbox_overlap_percentage(current_bbox, kept_bbox)
                
                if overlap_pct >= min_overlap_percentage:
                    # Current mask is a duplicate of a larger mask we're keeping
                    removed_masks.append(current_mask)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                masks_to_keep.append(current_mask)
        
        print(f"Removed {len(removed_masks)} duplicate masks, {len(masks_to_keep)} remaining")
        return masks_to_keep, removed_masks
    
    def _calculate_bbox_overlap_percentage(self, bbox1: List, bbox2: List) -> float:
        """Calculate the percentage of bbox1 that overlaps with bbox2 using bounding box coordinates."""
        # bbox format: [x, y, width, height]
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
    
    def create_huge_droplet_visualization(self, image: np.ndarray, final_masks: List[Dict], output_dir: str) -> str:
        """Create visualization showing all droplet masks with enhanced detail indication."""
        print(f"\nCreating huge droplet visualization...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(image)
        
        # Draw each mask with uniform color
        drawn_masks = 0
        uniform_color = (0.8, 0.2, 0.8)  # Magenta color for huge droplet mode
        
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
        
        # Create title with statistics and mode indication
        stats = self._calculate_droplet_statistics(final_masks)
        title = f"Huge Droplet Analysis (Enhanced SAM - crop_n_layers=3)\n(Total: {stats['mask_count']}, Avg Diameter: {stats['average_diameter']:.1f}px)"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"huge_droplet_analysis_{timestamp}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved huge droplet visualization: {output_path}")
        return output_path 
    
    def create_satellite_droplet_visualization(self, image: np.ndarray, group1_masks: List[Dict], group2_masks: List[Dict], output_dir: str) -> str:
        """Create visualization showing two groups of droplet masks with different colors and no labels."""
        print(f"\nCreating satellite droplet visualization...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(image)
        
        # Colors for the two groups
        group1_color = (0.2, 0.8, 0.2)  # Green for smaller masks (Group 1)
        group2_color = (0.8, 0.2, 0.2)  # Red for larger masks (Group 2)
        
        drawn_masks = 0
        
        # Draw Group 1 masks (smaller)
        for mask_data in group1_masks:
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
            
            # Draw bounding box for Group 1
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=group1_color, facecolor='none')
            ax.add_patch(rect)
            drawn_masks += 1
        
        # Draw Group 2 masks (larger)
        for mask_data in group2_masks:
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
            
            # Draw bounding box for Group 2
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=group2_color, facecolor='none')
            ax.add_patch(rect)
            drawn_masks += 1
        
        print(f"Successfully drew {drawn_masks} masks ({len(group1_masks)} Group 1 + {len(group2_masks)} Group 2)")
        
        # Create title with two-group statistics
        total_masks = len(group1_masks) + len(group2_masks)
        group1_stats = self._calculate_droplet_statistics(group1_masks) if group1_masks else {'mask_count': 0, 'average_diameter': 0}
        group2_stats = self._calculate_droplet_statistics(group2_masks) if group2_masks else {'mask_count': 0, 'average_diameter': 0}
        
        title = f"Satellite Droplet Analysis (K-means Two-Group Classification)\n"
        title += f"Group 1 (Green): {group1_stats['mask_count']} masks, Avg: {group1_stats['average_diameter']:.1f}px | "
        title += f"Group 2 (Red): {group2_stats['mask_count']} masks, Avg: {group2_stats['average_diameter']:.1f}px"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"satellite_droplet_analysis_{timestamp}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved satellite droplet visualization: {output_path}")
        return output_path 