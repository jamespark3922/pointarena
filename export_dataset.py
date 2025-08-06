#!/usr/bin/env python3
"""
Dataset export script for Point-Bench annotations.

This script exports annotation data to various formats including HuggingFace datasets,
COCO format, and other standard computer vision dataset formats.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import csv
import shutil
from PIL import Image
import numpy as np

def load_annotations(data_file="data.json"):
    """Load annotation data from JSON file."""
    try:
        with open(data_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {data_file} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {data_file}: {e}")
        return []

def export_huggingface(annotations, output_dir="pointbench_dataset"):
    """Export to HuggingFace dataset format."""
    print(f"📦 Exporting to HuggingFace format: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create directory structure
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "masks").mkdir(exist_ok=True)
    
    # Prepare dataset info
    dataset_info = {
        "dataset_name": "pointbench",
        "description": "Point-Bench: Multimodal Pointing Benchmark Dataset",
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "categories": ["affordable", "counting", "spatial", "reasoning", "steerable"],
        "total_samples": len(annotations)
    }
    
    # Process annotations
    hf_data = []
    for i, annotation in enumerate(annotations):
        # Copy image file
        image_filename = annotation.get("image_filename")
        if image_filename:
            src_image = find_image_file(image_filename)
            if src_image:
                dst_image = output_path / "images" / image_filename
                shutil.copy2(src_image, dst_image)
        
        # Copy mask file
        mask_filename = annotation.get("mask_filename")
        if mask_filename:
            src_mask = Path(mask_filename)
            if src_mask.exists():
                dst_mask = output_path / "masks" / src_mask.name
                shutil.copy2(src_mask, dst_mask)
                mask_filename = f"masks/{src_mask.name}"
        
        # Create HuggingFace format entry
        hf_entry = {
            "id": f"pointbench_{i:06d}",
            "image": f"images/{image_filename}" if image_filename else None,
            "mask": mask_filename,
            "query": annotation.get("user_input", ""),
            "category": annotation.get("category", ""),
            "count": annotation.get("count", 1),
            "timestamp": annotation.get("timestamp", "")
        }
        hf_data.append(hf_entry)
    
    # Save dataset
    with open(output_path / "dataset.json", "w") as f:
        json.dump(hf_data, f, indent=2)
    
    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create README
    readme_content = f"""# Point-Bench Dataset

## Overview
This dataset contains {len(annotations)} pointing task annotations across five categories:
- Affordable: Tool recognition tasks
- Counting: Object counting tasks  
- Spatial: Spatial relationship tasks
- Reasoning: Visual reasoning tasks
- Steerable: Reference-based tasks

## Format
- `dataset.json`: Main annotation file
- `images/`: Original images
- `masks/`: Binary mask annotations
- `dataset_info.json`: Dataset metadata

## Usage
```python
import json
with open('dataset.json', 'r') as f:
    data = json.load(f)
```

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_path / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ HuggingFace export complete: {len(hf_data)} samples")

def export_coco(annotations, output_dir="pointbench_coco"):
    """Export to COCO format."""
    print(f"📦 Exporting to COCO format: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create COCO structure
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "annotations").mkdir(exist_ok=True)
    
    # COCO annotation structure
    coco_data = {
        "info": {
            "description": "Point-Bench Dataset in COCO format",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create categories
    categories = ["affordable", "counting", "spatial", "reasoning", "steerable"]
    for i, category in enumerate(categories):
        coco_data["categories"].append({
            "id": i + 1,
            "name": category,
            "supercategory": "pointing_task"
        })
    
    # Process annotations
    image_id = 1
    annotation_id = 1
    
    for annotation in annotations:
        image_filename = annotation.get("image_filename")
        if not image_filename:
            continue
            
        # Copy image
        src_image = find_image_file(image_filename)
        if not src_image:
            continue
            
        dst_image = output_path / "images" / image_filename
        shutil.copy2(src_image, dst_image)
        
        # Get image dimensions
        try:
            with Image.open(src_image) as img:
                width, height = img.size
        except:
            continue
        
        # Add image entry
        coco_data["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_filename,
            "date_captured": annotation.get("timestamp", "")
        })
        
        # Add annotation entry
        category_name = annotation.get("category", "")
        category_id = categories.index(category_name) + 1 if category_name in categories else 1
        
        # Process mask if available
        bbox = [0, 0, width, height]  # Default to full image
        area = width * height
        segmentation = []
        
        mask_filename = annotation.get("mask_filename")
        if mask_filename and Path(mask_filename).exists():
            try:
                mask = Image.open(mask_filename)
                mask_array = np.array(mask)
                
                # Find bounding box of mask
                coords = np.where(mask_array > 0)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                    area = int(np.sum(mask_array > 0))
            except:
                pass
        
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
            "attributes": {
                "query": annotation.get("user_input", ""),
                "count": annotation.get("count", 1)
            }
        })
        
        image_id += 1
        annotation_id += 1
    
    # Save COCO annotation file
    with open(output_path / "annotations" / "instances_pointbench.json", "w") as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ COCO export complete: {len(coco_data['images'])} images")

def export_csv(annotations, output_file="pointbench_dataset.csv"):
    """Export to CSV format."""
    print(f"📦 Exporting to CSV format: {output_file}")
    
    fieldnames = ["id", "image_filename", "mask_filename", "query", "category", "count", "timestamp"]
    
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, annotation in enumerate(annotations):
            writer.writerow({
                "id": f"pointbench_{i:06d}",
                "image_filename": annotation.get("image_filename", ""),
                "mask_filename": annotation.get("mask_filename", ""),
                "query": annotation.get("user_input", ""),
                "category": annotation.get("category", ""),
                "count": annotation.get("count", 1),
                "timestamp": annotation.get("timestamp", "")
            })
    
    print(f"✅ CSV export complete: {len(annotations)} rows")

def find_image_file(filename):
    """Find image file in the standard directory structure."""
    # Check in images directory and subdirectories
    images_dir = Path("images")
    if not images_dir.exists():
        return None
    
    # Check in root images directory
    direct_path = images_dir / filename
    if direct_path.exists():
        return direct_path
    
    # Check in category subdirectories
    categories = ["affordable", "counting", "spatial", "reasoning", "steerable"]
    for category in categories:
        category_path = images_dir / category / filename
        if category_path.exists():
            return category_path
    
    return None

def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export Point-Bench dataset to various formats")
    parser.add_argument("--format", choices=["huggingface", "coco", "csv", "all"], 
                       default="huggingface", help="Export format")
    parser.add_argument("--output", default=None, help="Output directory/file")
    parser.add_argument("--data", default="data.json", help="Input annotation file")
    
    args = parser.parse_args()
    
    print("📤 Point-Bench Dataset Export")
    print("=" * 30)
    
    # Load annotations
    annotations = load_annotations(args.data)
    if not annotations:
        print("❌ No annotations found or failed to load data file")
        return False
    
    print(f"📊 Found {len(annotations)} annotations")
    
    # Export based on format
    if args.format == "huggingface" or args.format == "all":
        output_dir = args.output or "pointbench_dataset"
        export_huggingface(annotations, output_dir)
    
    if args.format == "coco" or args.format == "all":
        output_dir = args.output or "pointbench_coco"
        export_coco(annotations, output_dir)
    
    if args.format == "csv" or args.format == "all":
        output_file = args.output or "pointbench_dataset.csv"
        export_csv(annotations, output_file)
    
    print("🏁 Export complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)