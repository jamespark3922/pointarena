#!/usr/bin/env python3
"""
Dataset validation script for Point-Bench annotations.

This script validates the structure and integrity of annotation data,
checking for missing files, invalid formats, and data consistency issues.
"""

import json
import sys
from pathlib import Path
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

def validate_annotation(annotation_data, index):
    """Validate a single annotation entry."""
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ["image_filename", "user_input", "category", "timestamp"]
    for field in required_fields:
        if field not in annotation_data:
            errors.append(f"Missing required field: {field}")
    
    # Optional but recommended fields
    recommended_fields = ["mask_filename", "count"]
    for field in recommended_fields:
        if field not in annotation_data:
            warnings.append(f"Missing recommended field: {field}")
    
    # Validate category
    valid_categories = ["affordable", "counting", "spatial", "reasoning", "steerable"]
    if "category" in annotation_data:
        if annotation_data["category"] not in valid_categories:
            errors.append(f"Invalid category: {annotation_data['category']}")
    
    # Check image file exists
    if "image_filename" in annotation_data:
        image_path = find_image_file(annotation_data["image_filename"])
        if not image_path:
            errors.append(f"Image file not found: {annotation_data['image_filename']}")
    
    # Check mask file exists
    if "mask_filename" in annotation_data:
        mask_path = Path(annotation_data["mask_filename"])
        if not mask_path.exists():
            errors.append(f"Mask file not found: {mask_path}")
        else:
            # Validate mask format
            try:
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                if len(mask_array.shape) != 2:
                    warnings.append(f"Mask should be grayscale: {mask_path}")
                if mask_array.dtype != np.uint8:
                    warnings.append(f"Mask should be uint8 format: {mask_path}")
            except Exception as e:
                errors.append(f"Cannot load mask image {mask_path}: {e}")
    
    # Validate count field for counting category
    if annotation_data.get("category") == "counting":
        if "count" not in annotation_data:
            errors.append("Counting category requires 'count' field")
        elif not isinstance(annotation_data["count"], int) or annotation_data["count"] < 1:
            errors.append("Count must be a positive integer")
    
    # Validate user input
    if "user_input" in annotation_data:
        if not annotation_data["user_input"].strip():
            errors.append("User input cannot be empty")
        if len(annotation_data["user_input"]) > 500:
            warnings.append("User input is very long (>500 characters)")
    
    return errors, warnings

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

def generate_statistics(annotations):
    """Generate dataset statistics."""
    stats = {
        "total_annotations": len(annotations),
        "categories": {},
        "missing_masks": 0,
        "missing_images": 0,
        "counting_tasks": 0,
        "average_query_length": 0
    }
    
    query_lengths = []
    
    for annotation in annotations:
        # Category distribution
        category = annotation.get("category", "unknown")
        stats["categories"][category] = stats["categories"].get(category, 0) + 1
        
        # Missing files
        if "mask_filename" not in annotation or not Path(annotation["mask_filename"]).exists():
            stats["missing_masks"] += 1
        
        if not find_image_file(annotation.get("image_filename", "")):
            stats["missing_images"] += 1
        
        # Counting tasks
        if category == "counting":
            stats["counting_tasks"] += 1
        
        # Query length
        if "user_input" in annotation:
            query_lengths.append(len(annotation["user_input"]))
    
    if query_lengths:
        stats["average_query_length"] = sum(query_lengths) / len(query_lengths)
    
    return stats

def main():
    """Main validation function."""
    print("🔍 Point-Bench Dataset Validation")
    print("=" * 40)
    
    # Load annotations
    annotations = load_annotations()
    if not annotations:
        print("❌ No annotations found or failed to load data.json")
        return False
    
    print(f"📊 Found {len(annotations)} annotations")
    
    # Validate each annotation
    total_errors = 0
    total_warnings = 0
    
    for i, annotation in enumerate(annotations):
        errors, warnings = validate_annotation(annotation, i)
        
        if errors:
            print(f"\n❌ Annotation {i+1} has {len(errors)} error(s):")
            for error in errors:
                print(f"   • {error}")
            total_errors += len(errors)
        
        if warnings:
            print(f"\n⚠️  Annotation {i+1} has {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"   • {warning}")
            total_warnings += len(warnings)
    
    # Generate statistics
    stats = generate_statistics(annotations)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total annotations: {stats['total_annotations']}")
    print(f"   Missing masks: {stats['missing_masks']}")
    print(f"   Missing images: {stats['missing_images']}")
    print(f"   Counting tasks: {stats['counting_tasks']}")
    print(f"   Average query length: {stats['average_query_length']:.1f} characters")
    
    print(f"\n📂 Category distribution:")
    for category, count in stats['categories'].items():
        percentage = (count / stats['total_annotations']) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    # Summary
    print(f"\n🏁 Validation Summary:")
    print(f"   Total errors: {total_errors}")
    print(f"   Total warnings: {total_warnings}")
    
    if total_errors == 0:
        print("✅ Dataset validation passed!")
        return True
    else:
        print("❌ Dataset validation failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)