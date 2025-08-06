#!/usr/bin/env python3
"""
Batch annotation script for Point-Bench.

This script provides utilities for batch processing images and generating
annotations using various automated methods including SAM integration.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import torch

# Import SAM if available
try:
    from segment_utils import SegmentAnythingHelper
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: SAM not available. Install segment-anything for automatic segmentation.")
    SAM_AVAILABLE = False

def load_existing_annotations(data_file="data.json"):
    """Load existing annotations to avoid duplicates."""
    try:
        with open(data_file, "r") as f:
            annotations = json.load(f)
        return {ann.get("image_filename"): ann for ann in annotations}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_annotation(annotation_data, data_file="data.json"):
    """Save a single annotation to the data file."""
    # Load existing data
    try:
        with open(data_file, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    # Add timestamp
    annotation_data["timestamp"] = datetime.now().isoformat()
    
    # Append new annotation
    existing_data.append(annotation_data)
    
    # Save back to file
    with open(data_file, "w") as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"✅ Saved annotation for {annotation_data.get('image_filename', 'unknown')}")

def save_mask_image(mask, original_image_path, masks_dir="masks"):
    """Save a mask image and return the filename."""
    masks_path = Path(masks_dir)
    masks_path.mkdir(exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_stem = Path(original_image_path).stem
    mask_filename = f"{image_stem}_mask_{timestamp}.png"
    mask_path = masks_path / mask_filename
    
    # Convert mask to PIL Image and save
    if isinstance(mask, np.ndarray):
        # Convert boolean mask to uint8
        if mask.dtype == bool:
            mask = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask, mode='L')
    else:
        mask_image = mask
    
    mask_image.save(mask_path)
    
    return str(mask_path)

def generate_sam_annotation(image_path, query, category, sam_helper):
    """Generate annotation using SAM automatic segmentation."""
    if not SAM_AVAILABLE:
        print("❌ SAM not available for automatic annotation")
        return None
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # For now, use center point as initial prompt
        # In a more sophisticated version, this could use text-to-point models
        width, height = image.size
        center_points = [(width // 2, height // 2)]
        
        # Generate mask using SAM
        mask = sam_helper.segment_with_points(image, center_points)
        
        if mask is not None:
            # Save mask
            mask_filename = save_mask_image(mask, image_path)
            
            # Create annotation
            annotation = {
                "image_filename": Path(image_path).name,
                "user_input": query,
                "category": category,
                "mask_filename": mask_filename,
                "count": 1 if category != "counting" else estimate_object_count(mask),
                "method": "sam_auto"
            }
            
            return annotation
    
    except Exception as e:
        print(f"❌ Error generating SAM annotation for {image_path}: {e}")
    
    return None

def estimate_object_count(mask):
    """Estimate number of objects in a mask using connected components."""
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    # Simple connected components analysis
    from scipy import ndimage
    labeled_mask, num_features = ndimage.label(mask > 0)
    return max(1, num_features)

def generate_manual_template(image_path, query, category):
    """Generate a template annotation for manual completion."""
    annotation = {
        "image_filename": Path(image_path).name,
        "user_input": query,
        "category": category,
        "mask_filename": "",  # To be filled manually
        "count": 1 if category != "counting" else 0,  # To be filled manually
        "method": "manual_template"
    }
    
    return annotation

def batch_process_directory(image_dir, category, queries=None, method="template", sam_helper=None):
    """Process all images in a directory."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"❌ Directory not found: {image_dir}")
        return
    
    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in image_dir.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No image files found in {image_dir}")
        return
    
    print(f"🔄 Processing {len(image_files)} images in {image_dir}")
    
    # Load existing annotations to avoid duplicates
    existing = load_existing_annotations()
    
    # Process each image
    processed = 0
    skipped = 0
    
    for image_file in image_files:
        # Skip if already annotated
        if image_file.name in existing:
            print(f"⏭️  Skipping {image_file.name} (already annotated)")
            skipped += 1
            continue
        
        # Generate query for this image
        if queries and isinstance(queries, list):
            query = queries[processed % len(queries)]
        elif queries and isinstance(queries, str):
            query = queries
        else:
            query = f"Point to the main object in this {category} image"
        
        # Generate annotation based on method
        annotation = None
        
        if method == "sam" and sam_helper:
            annotation = generate_sam_annotation(image_file, query, category, sam_helper)
        elif method == "template":
            annotation = generate_manual_template(image_file, query, category)
        
        if annotation:
            save_annotation(annotation)
            processed += 1
        else:
            print(f"❌ Failed to process {image_file.name}")
    
    print(f"✅ Processed {processed} images, skipped {skipped}")

def interactive_annotation(image_path, sam_helper=None):
    """Interactive annotation for a single image."""
    print(f"🖼️  Annotating: {image_path}")
    
    # Get user input
    query = input("Enter pointing query: ").strip()
    if not query:
        print("❌ Empty query, skipping")
        return
    
    print("Categories: affordable, counting, spatial, reasoning, steerable")
    category = input("Enter category: ").strip().lower()
    if category not in ["affordable", "counting", "spatial", "reasoning", "steerable"]:
        print("❌ Invalid category, skipping")
        return
    
    # Get count for counting tasks
    count = 1
    if category == "counting":
        try:
            count = int(input("Enter object count: "))
        except ValueError:
            count = 1
    
    # Choose annotation method
    if SAM_AVAILABLE and sam_helper:
        method = input("Use SAM for automatic segmentation? (y/n): ").strip().lower()
        if method == "y":
            annotation = generate_sam_annotation(image_path, query, category, sam_helper)
            if annotation:
                annotation["count"] = count
                save_annotation(annotation)
                return
    
    # Manual template
    annotation = generate_manual_template(image_path, query, category)
    annotation["count"] = count
    save_annotation(annotation)
    print("📝 Template saved. Complete annotation using the web interface.")

def main():
    """Main function for batch annotation."""
    parser = argparse.ArgumentParser(description="Batch annotation for Point-Bench")
    parser.add_argument("--mode", choices=["batch", "interactive"], default="batch",
                       help="Annotation mode")
    parser.add_argument("--input", required=True, help="Input directory or image file")
    parser.add_argument("--category", help="Image category")
    parser.add_argument("--query", help="Pointing query (for batch mode)")
    parser.add_argument("--method", choices=["template", "sam"], default="template",
                       help="Annotation method")
    parser.add_argument("--sam-checkpoint", help="Path to SAM checkpoint")
    
    args = parser.parse_args()
    
    print("🤖 Point-Bench Batch Annotation")
    print("=" * 35)
    
    # Initialize SAM if requested
    sam_helper = None
    if args.method == "sam" or args.mode == "interactive":
        if SAM_AVAILABLE and args.sam_checkpoint and Path(args.sam_checkpoint).exists():
            try:
                sam_helper = SegmentAnythingHelper(
                    checkpoint_path=args.sam_checkpoint,
                    model_type="vit_h"
                )
                print("✅ SAM initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize SAM: {e}")
                if args.method == "sam":
                    return
        elif args.method == "sam":
            print("❌ SAM checkpoint required for SAM method")
            return
    
    # Process based on mode
    input_path = Path(args.input)
    
    if args.mode == "batch":
        if input_path.is_dir():
            if not args.category:
                # Try to infer category from directory name
                category = input_path.name.lower()
                if category not in ["affordable", "counting", "spatial", "reasoning", "steerable"]:
                    print("❌ Category required for batch processing")
                    return
            else:
                category = args.category
            
            batch_process_directory(
                input_path, 
                category, 
                queries=args.query,
                method=args.method,
                sam_helper=sam_helper
            )
        else:
            print("❌ Batch mode requires a directory")
    
    elif args.mode == "interactive":
        if input_path.is_file():
            interactive_annotation(input_path, sam_helper)
        else:
            print("❌ Interactive mode requires a single image file")

if __name__ == "__main__":
    main()