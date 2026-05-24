<p align="center">
    <h1 align="center">
    <img src="https://pointarena.github.io/favicon.ico" width="25px"/>
        PointArena: Probing Multimodal Grounding Through Language-Guided Pointing
    </h1>
</p>

<p align="center">
  <a href="https://victorthecreator.me/">Long Cheng<sup>1∗</sup></a>, 
  <a href="https://duanjiafei.com">Jiafei Duan<sup>1,2∗</sup></a>, 
  <a href="https://helen9975.github.io">Yi Ru Wang<sup>1†</sup></a>, 
  <a href="https://hq-fang.github.io">Haoquan Fang<sup>1,2†</sup></a>, 
  <a href="#">Boyang Li<sup>1†</sup></a>, 
  <br>
  <a href="#">Yushan Huang<sup>1</sup></a>, 
  <a href="#">Elvis Wang<sup>3</sup></a>, 
  <a href="#">Ainaz Eftekhar<sup>1,2</sup></a>, 
  <a href="#">Jason Lee<sup>1,2</sup></a>, 
  <a href="#">Wentao Yuan<sup>1</sup></a>, 
  <br>
  <a href="#">Rose Hendrix<sup>2</sup></a>, 
  <a href="https://nasmith.github.io/">Noah A. Smith<sup>1,2</sup></a>, 
  <a href="https://linguistics.washington.edu/people/fei-xia">Fei Xia<sup>1</sup></a>, 
  <a href="https://homes.cs.washington.edu/~fox">Dieter Fox<sup>1</sup></a>, 
  <a href="https://ranjaykrishna.com">Ranjay Krishna<sup>1,2</sup></a>
  <br><br>
  <sup>1</sup>University of Washington, 
  <sup>2</sup>Allen Institute for Artificial Intelligence, 
  <sup>3</sup>Anderson Collegiate Vocational Institute
  <br>
∗Co-first authors.
†Co-second authors.
  
</p>

<div align="center">
  <p>
    <a href="https://pointarena.github.io/">
      <img src="https://img.shields.io/badge/Website-grey?logo=google-chrome&logoColor=white&labelColor=blue">
    </a>
    <a href="https://arxiv.org/abs/2505.09990">
      <img src="https://img.shields.io/badge/arXiv-grey?logo=arxiv&logoColor=white&labelColor=red">
    </a>
    <a href="https://huggingface.co/datasets/PointArena/pointarena-data">
      <img src="https://img.shields.io/badge/Dataset-grey?logo=huggingface&logoColor=white&labelColor=yellow">
    </a>
    <a href="https://x.com/victor_UWer">
      <img src="https://img.shields.io/badge/Post-grey?logo=x&logoColor=white&labelColor=black">
    </a>
  </p>
</div>


Pointing serves as a fundamental and intuitive mechanism for grounding language within visual contexts, with applications spanning robotics, assistive technologies, and interactive AI systems. While recent multimodal models have begun supporting pointing capabilities, existing benchmarks typically focus only on referential object localization. We introduce PointArena, a comprehensive platform for evaluating multimodal pointing across diverse reasoning scenarios. PointArena comprises three components: (1) Point-Bench, a curated dataset of approximately 1,000 pointing tasks across five reasoning categories; (2) Point-Battle, an interactive web-based arena facilitating blind, pairwise model comparisons, which has collected over 4,500 anonymized votes; and (3) Point-Act, a real-world robotic manipulation system allowing users to directly evaluate model pointing in practical settings. We conducted extensive evaluations of both state-of-the-art open-source and proprietary models. Results indicate that Molmo-72B consistently outperforms others, though proprietary models increasingly demonstrate comparable performance. Additionally, we find that supervised training targeting pointing tasks significantly improves performance. Across our multi-stage evaluation pipeline, we observe strong correlations, underscoring the critical role of precise pointing in enabling multimodal models to bridge abstract reasoning with real-world actions.


## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Static Evaluation Interface](#static-evaluation-interface)
  - [Point-Bench](#point-bench)
  - [Point-Battle](#point-battle)
  - [Performance Analysis](#performance-analysis)
- [Data Collection and Annotation Guide](#data-collection-and-annotation-guide)
  - [Annotation System Architecture](#annotation-system-architecture)
  - [Data Format and Structure](#data-format-and-structure)
  - [Setting Up Your Annotation Environment](#setting-up-your-annotation-environment)
  - [Annotation Workflow](#annotation-workflow)
  - [Scaling Up Point-Bench](#scaling-up-point-bench)
  - [Integration with Existing Datasets](#integration-with-existing-datasets)
  - [Performance Optimization](#performance-optimization)
  - [Monitoring and Analytics](#monitoring-and-analytics)
  - [Troubleshooting Common Issues](#troubleshooting-common-issues)
- [Project Structure](#project-structure)
- [Image Categories](#image-categories)
- [Model Support](#model-support)
- [Data and Evaluation](#data-and-evaluation)
- [Requirements](#requirements)

## Key Features

- **Annotation System**: Grid-based selection interface for precise point annotations
- **Segment Anything Model (SAM) Integration**: Automatic segmentation using Meta's Segment Anything Model
- **Multi-Model Evaluation**: Compare various vision-language models including:
  - OpenAI models (GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano)
  - Google models (Gemini 2.5/2.0 series, including flash and pro variants)
  - Open-source models (Molmo series, Qwen 2.5-VL, LLaVA OneVision)
  - Claude (claude-3-7-sonnet-20250219) and Grok (grok-2-vision-latest) models
- **Performance Analysis**: Visualize model performance with:
  - ELO ratings system with confidence intervals
  - Pairwise win rates and match count heatmaps
  - Success rate metrics and performance summaries
- **Dynamic Testing Mode**: Test models in real-time with user-uploaded images
- **Human Benchmark**: Compare model performance against human baselines
- **Comprehensive Annotation Tools**: Batch processing, validation, and export utilities

## Installation

### Core System

1. Clone the repository:
```bash 
git clone <repository-url>
cd pointarena
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Molmo model evaluation:
```bash
pip install -r requirements_molmo.txt
```

4. Download the PointArena benchmark data:
```bash
pip install huggingface_hub
./scripts/download_pointarena_data.sh
```

This creates the files and folders required by `molmo_evaluator.py`:

```text
data.json
pixmo_metadata.csv
selected_images/
selected_masks/
```

5. Create a `.env` file with your API keys:
```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your API keys and settings
# Required keys:
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
XAI_API_KEY=your_xai_api_key
SAM_CHECKPOINT_PATH=./sam_vit_h_4b8939.pth
SAM_MODEL_TYPE=vit_h
SAVED_MODELS_DIR=./models
```

6. Download the SAM model checkpoint:
```bash
# Download directly from Meta AI's repository
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

### Static Evaluation Interface

1. Start the annotation interface:
```bash
python app.py
```

2. Open your browser at `http://localhost:7860`

3. Use the interface to:
   - Manually annotate images with grid selection
   - Use SAM for automatic object segmentation
   - Compare different model predictions
   - Save annotations to a structured data format


### Point-Bench

Evaluate vision-language models on point recognition tasks:

```bash
# Run evaluation for a specific model
# For example:
python model_evaluator.py --model gpt-4o --type openai
python model_evaluator.py --model gemini-2.0-flash --type gemini
python molmo_evaluator.py --model Molmo-7B-D-0924 --type molmo
```

For MolmoPoint-8B and Molmo2-8B:

```bash
./scripts/run_molmopoint_8b.sh --no-resume
./scripts/run_molmo2_8b.sh --no-resume
```

Use `--no-resume` when changing prompts, point parsing, model checkpoints, or after a failed run. Otherwise the evaluator may reuse entries from `static_results/`.

The evaluator will:
1. Generate visualizations showing points predicted by each model
2. Save these visualizations to the `point_on_mask` directory
3. Create a JSON results file with detailed metrics   

MolmoPoint/Molmo2 outputs are saved to:

```text
static_results/results_molmo_allenai_MolmoPoint-8B_simple_prompt.json
static_results/results_molmo_allenai_Molmo2-8B_simple_prompt.json
```

Current runs print and save a leaderboard-style category breakdown with:

```text
Affordance | Spatial | Reasoning | Steerability | Counting | Average
```

For an older, renamed, or copied result file that does not already have a category summary, run:

```bash
python scripts/summarize_pointarena_results.py \
  static_results/results_molmo_allenai_MolmoPoint-8B_simple_prompt-v1.json
```


### Point-Battle

1. Start the dynamic testing interface:
```bash
python dynamic.py
```

2. Open your browser at `http://localhost:7860`

3. Use the interface to:
   - Test models with provided test images from different categories
   - Upload your own images for testing
   - Compare model performance in head-to-head battles
   - View dynamic ELO leaderboard



### Performance Analysis

Generate performance visualizations and statistics:

```bash
# Generate ELO leaderboard with confidence intervals
python elo_leaderboard.py

# Generate pairwise win rates and match counts
python pairwise_win_rates.py

# For human benchmark comparison
python human_benchmark.py
```

### Data Management Utilities

Validate, export, and batch process your annotation data:

```bash
# Validate dataset integrity and structure
python validate_dataset.py

# Export to different formats
python export_dataset.py --format huggingface
python export_dataset.py --format coco --output pointbench_coco
python export_dataset.py --format csv --output dataset.csv

# Batch annotation processing
python batch_annotate.py --mode batch --input images/counting --category counting
python batch_annotate.py --mode interactive --input image.jpg
```

## Project Structure

### Core Components
- `app.py`: Main annotation application with Gradio UI for static evaluation
- `dynamic.py`: Point-Battle interface for head-to-head model comparisons
- `model_evaluator.py`: Point-Bench interface for evaluating different vision-language models
- `molmo_evaluator.py`: Point-Bench interface for evaluating Molmo models specifically
- `elo_leaderboard.py`: Generate ELO ratings and confidence intervals for model performance
- `pairwise_win_rates.py`: Calculate and visualize pairwise model comparisons with heatmaps
- `human_benchmark.py`: Evaluate human performance baseline

### API and Model Support
- `molmo_api.py`: API client for Molmo model inference with support for local or remote execution
- `optimize_user_input.py`: Optimize user prompts for better model performance
- `segment_utils.py`: Helper utilities for the Segment Anything Model integration

### Data Management and Utilities
- `validate_dataset.py`: Comprehensive dataset validation and quality control
- `export_dataset.py`: Export annotations to various formats (HuggingFace, COCO, CSV)
- `batch_annotate.py`: Batch processing and automated annotation tools
- `.env.example`: Configuration template for easy setup

## Image Categories

The system supports five specialized task categories:
1. **Affordable**: Tool recognition tasks requiring fine-grained object identification
2. **Counting**: Object counting tasks with numerical reasoning requirements
3. **Spatial**: Spatial relationship tasks requiring positional understanding
4. **Reasoning**: Visual reasoning tasks requiring complex visual inference
5. **Steerable**: Tasks with reference points requiring contextual understanding

## Model Support

### OpenAI Models
- gpt-4o
- o3
- gpt-4.1

### Google Models
- gemini-2.5-flash-preview-04-17
- gemini-2.5-pro-preview-05-06
- gemini-2.0-flash

### Open Source Models
- MolmoPoint-8B
- Molmo2-8B
- Molmo-7B-D-0924
- Molmo-7B-O-0924
- Molmo-72B-0924
- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-32B-Instruct
- Qwen2.5-VL-72B-Instruct
- llava-onevision-qwen2-7b-ov-hf

### Additional Models
- claude-3-7-sonnet-20250219
- grok-2-vision-latest

## Data and Evaluation

- Uses a structured annotation format with point coordinates
- Stores masked regions for precise evaluation
- Supports multiple evaluation metrics:
  - Point-in-mask accuracy
  - ELO rating system with confidence intervals
  - Pairwise win rate comparisons
  - Total success rate across categories

## Data Collection and Annotation Guide

### Overview

Point-Bench uses a structured annotation system for collecting pointing data across five task categories. This section provides comprehensive instructions for collecting your own data and scaling up the benchmark.

### Annotation System Architecture

The annotation system consists of three main components:

1. **Grid-based Manual Selection**: 50×50 grid overlay for precise point selection
2. **SAM Integration**: Automatic segmentation using Meta's Segment Anything Model
3. **Structured Data Storage**: JSON format with associated mask images

### Data Format and Structure

#### Annotation Data Format (`data.json`)

Each annotation entry contains the following structure:

```json
{
  "image_filename": "example_001.jpg",
  "user_input": "red apple on the table",
  "category": "spatial",
  "count": 1,
  "mask_filename": "masks/example_001_mask_20240315_143022.png",
  "timestamp": "2024-03-15T14:30:22.123456"
}
```

**Field Descriptions:**
- `image_filename`: Original image file name
- `user_input`: The pointing query/instruction 
- `category`: One of ["affordable", "counting", "spatial", "reasoning", "steerable"]
- `count`: Number of objects to point to (important for counting tasks)
- `mask_filename`: Path to the corresponding mask image
- `timestamp`: ISO format timestamp of annotation creation

#### Mask Image Format

- **Format**: PNG images stored in `masks/` directory
- **Naming**: `{original_filename}_mask_{timestamp}.png`
- **Content**: Binary masks where white pixels (255) indicate target regions
- **Coordinates**: Match original image dimensions

### Setting Up Your Annotation Environment

#### 1. Prepare Your Image Collection

```bash
# Create directory structure
mkdir -p images/{affordable,counting,spatial,reasoning,steerable}

# Place your images in appropriate category folders
# - affordable: Tool recognition, fine-grained object identification
# - counting: Object counting tasks with numerical reasoning
# - spatial: Positional relationships and spatial understanding
# - reasoning: Complex visual inference tasks
# - steerable: Tasks with reference points requiring context
```

#### 2. Configure Environment

```bash
# Set up environment variables
cp .env.example .env

# Edit .env with your configurations:
SAM_CHECKPOINT_PATH=./sam_vit_h_4b8939.pth
SAM_MODEL_TYPE=vit_h
SAVED_MODELS_DIR=./models
```

#### 3. Download SAM Model

```bash
# Download the Segment Anything Model checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Annotation Workflow

#### Method 1: Manual Grid-Based Annotation

1. **Start the Annotation Interface**:
   ```bash
   python app.py
   ```

2. **Access the Interface**: Open `http://localhost:7860` in your browser

3. **Annotation Process**:
   - Navigate to the "Manual Annotation" tab
   - Enter your pointing query in the text box
   - Click "Submit Test" to load a random image
   - Use the grid overlay to select target regions by clicking cells
   - For counting tasks, use the counter to specify number of objects
   - Click "Complete Manual Annotation" to save

4. **Grid Selection Tips**:
   - Select multiple cells to cover larger objects
   - Use Shift+Click for multi-selection
   - The 50×50 grid provides high precision
   - Ensure complete coverage of target objects

#### Method 2: SAM-Assisted Annotation

1. **Enable SAM Mode**: In the annotation interface, enable "Use SAM"

2. **SAM Annotation Process**:
   - Click on the object you want to segment
   - SAM will automatically generate a precise mask
   - Review the generated mask for accuracy
   - Adjust by clicking additional points if needed
   - Accept the mask and save the annotation

3. **SAM Benefits**:
   - More precise object boundaries
   - Faster annotation for complex shapes
   - Consistent quality across annotators

### Scaling Up Point-Bench

#### 1. Distributed Annotation Setup

For large-scale data collection, set up multiple annotation stations:

```bash
# Set up multiple annotation instances
# Station 1: Port 7860 (default)
python app.py

# Station 2: Port 7861
python app.py --port 7861

# Station 3: Port 7862
python app.py --port 7862
```

#### 2. Quality Control Framework

Implement quality control measures:

```python
# Example quality control script
def validate_annotation(annotation_data):
    required_fields = ["image_filename", "user_input", "category", "mask_filename"]
    
    # Check required fields
    for field in required_fields:
        if field not in annotation_data:
            return False, f"Missing required field: {field}"
    
    # Validate category
    valid_categories = ["affordable", "counting", "spatial", "reasoning", "steerable"]
    if annotation_data["category"] not in valid_categories:
        return False, f"Invalid category: {annotation_data['category']}"
    
    # Check mask file exists
    mask_path = Path(annotation_data["mask_filename"])
    if not mask_path.exists():
        return False, f"Mask file not found: {mask_path}"
    
    return True, "Valid annotation"
```

#### 3. Annotation Guidelines

**General Principles:**
- **Precision**: Select exact object boundaries, not approximate regions
- **Consistency**: Use consistent criteria across similar tasks
- **Completeness**: Ensure all relevant objects are marked for counting tasks
- **Context**: Consider spatial relationships and contextual clues

**Category-Specific Guidelines:**

**Affordable Tasks:**
- Focus on specific tools or objects mentioned in the query
- Pay attention to fine-grained details (e.g., specific tool types)
- Distinguish between similar objects

**Counting Tasks:**
- Mark each individual object instance
- Use the counter to specify exact number
- Ensure no objects are double-counted or missed

**Spatial Tasks:**
- Consider relative positions (left, right, above, below)
- Account for spatial prepositions in queries
- Mark objects in context of their relationships

**Reasoning Tasks:**
- Analyze the visual scene for implicit information
- Consider object properties, states, and conditions
- Focus on inferential aspects of the query

**Steerable Tasks:**
- Use reference points when provided
- Consider directional and contextual cues
- Adapt annotation based on steering context

#### 4. Batch Processing and Automation

For processing large image datasets:

```python
# Example batch annotation script
import os
from pathlib import Path

def batch_process_images(image_directory, category):
    """Process images in batch for a specific category."""
    image_paths = list(Path(image_directory).glob("*.jpg"))
    
    for image_path in image_paths:
        # Load image
        # Generate automatic annotations using SAM
        # Save to data.json
        pass

# Usage
batch_process_images("images/counting", "counting")
```

#### 5. Data Validation and Export

Validate your dataset before using it for evaluation:

```bash
# Run validation script
python validate_dataset.py

# Export to standard format
python export_dataset.py --format huggingface
```

### Integration with Existing Datasets

#### Converting External Datasets

To integrate datasets from other sources:

1. **Prepare Conversion Script**:
   ```python
   def convert_external_dataset(source_path, target_format="pointbench"):
       # Load external dataset
       # Convert to Point-Bench format
       # Generate masks if needed
       # Save to data.json
       pass
   ```

2. **Standardize Categories**: Map external categories to Point-Bench categories

3. **Generate Masks**: Use SAM to generate masks from bounding boxes or points

#### Metadata Management

Track dataset metadata for better organization:

```python
# metadata.json structure
{
    "dataset_version": "1.0",
    "creation_date": "2024-03-15",
    "total_annotations": 1000,
    "category_distribution": {
        "affordable": 200,
        "counting": 200,
        "spatial": 200,
        "reasoning": 200,
        "steerable": 200
    },
    "annotators": ["annotator_1", "annotator_2"],
    "quality_metrics": {
        "inter_annotator_agreement": 0.85,
        "average_annotation_time": 45.2
    }
}
```

### Performance Optimization

#### 1. Annotation Speed Optimization

- **Keyboard Shortcuts**: Implement hotkeys for common actions
- **Batch Operations**: Allow selection of multiple cells at once
- **Smart Defaults**: Use previous annotations to suggest defaults
- **Undo/Redo**: Implement undo functionality for quick corrections

#### 2. Storage Optimization

```python
# Compress masks for storage efficiency
def compress_mask(mask_path):
    """Compress mask images to reduce storage."""
    # Use PNG compression or convert to run-length encoding
    pass

# Database integration for large-scale deployments
def setup_database():
    """Set up database for annotation storage."""
    # SQLite for local, PostgreSQL for distributed
    pass
```

### Monitoring and Analytics

Track annotation progress and quality:

```python
def generate_annotation_report():
    """Generate progress and quality report."""
    return {
        "annotations_per_day": calculate_daily_rate(),
        "category_completion": calculate_category_progress(),
        "quality_metrics": calculate_quality_scores(),
        "annotator_performance": calculate_annotator_stats()
    }
```

### Troubleshooting Common Issues

#### 1. SAM Integration Issues
- **GPU Memory**: Reduce batch size for SAM processing
- **Model Loading**: Ensure checkpoint path is correct
- **CUDA Issues**: Verify PyTorch CUDA installation

#### 2. Data Consistency Issues
- **File Paths**: Use absolute paths for mask files
- **JSON Format**: Validate JSON structure before saving
- **Image Formats**: Ensure consistent image formats (JPG/PNG)

#### 3. Performance Issues
- **Large Images**: Resize images for faster processing
- **Memory Usage**: Clear cached data periodically
- **Concurrent Access**: Use file locking for multi-user setups

This comprehensive guide provides the foundation for scaling Point-Bench to larger datasets while maintaining annotation quality and consistency.

## Requirements

Core dependencies:
- PyTorch (2.2.0) and torchvision (0.17.0)
- Gradio (5.22.0) for interactive interfaces
- OpenAI, Google Generative AI, Anthropic, and x.ai APIs
- Segment Anything Model from Meta AI
- Transformers library for local model inference
- Pillow, NumPy, Matplotlib for image processing and visualization
- FastAPI and Uvicorn for API services
- Pandas and Seaborn for data analysis and visualization
