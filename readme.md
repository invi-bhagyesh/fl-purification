# FL-Purification: Data Generation and Training/Testing Guide

This guide provides the essential commands for data generation, loading, and running training/testing for the FL-Purification project.

## Quick Start

### 1. Data Generation (Kaggle)

```bash
# Generate dataset with all attack types and strengths
python src/run.py --mode prepare_data --kaggle_mode --dataset_name bloodmnist --create_kaggle_dataset
```

### 2. Training

```bash
# Train detector (autoencoder)
!python src/run.py --mode train --train_clean_only --kaggle_mode --model_type detector --dataset_name bloodmnist --attack_type fgsm --attack_strength strong

# Train reformer
!python src/run.py --mode train --train_clean_only --kaggle_mode --model_type reformer --dataset_name bloodmnist --attack_type fgsm --attack_strength strong

# Train classifier
!python src/run.py --mode train --train_clean_only --kaggle_mode --model_type classifier --dataset_name bloodmnist --attack_type fgsm --attack_strength strong

# Train all models (pipeline)
!python src/run.py --mode train --train_clean_only --kaggle_mode --model_type all --dataset_name bloodmnist --attack_type fgsm --attack_strength strong

# Train classifier on clean bloodmnist data
!python src/run.py --mode train --model_type classifier --dataset_name bloodmnist

# Train full pipeline on clean dermamnist data  
!python src/run.py --mode train --model_type all --dataset_name dermamnist --epochs 30
```

### 3. Testing

```bash
# Test individual model
python src/run.py --mode test --kaggle_mode --test_type individual --model_type detector --dataset_name bloodmnist --attack_type fgsm --attack_strength medium

# Test pipeline configurations
python src/run.py --mode test --kaggle_mode --test_type pipeline --dataset_name bloodmnist --attack_type fgsm --attack_strength medium
```

## Data Generation Commands

### Single Dataset
```bash
python src/run.py --mode prepare_data --kaggle_mode --dataset_name bloodmnist --create_kaggle_dataset
```

### Multiple Datasets
```bash
# Generate for different datasets
python src/run.py --mode prepare_data --kaggle_mode --dataset_name pathmnist --create_kaggle_dataset
python src/run.py --mode prepare_data --kaggle_mode --dataset_name dermamnist --create_kaggle_dataset
python src/run.py --mode prepare_data --kaggle_mode --dataset_name octmnist --create_kaggle_dataset
```

### Custom Settings
```bash
# With custom attack strength
python src/run.py --mode prepare_data --kaggle_mode --dataset_name bloodmnist --attack_strength strong --create_kaggle_dataset

# With specific device
python src/run.py --mode prepare_data --kaggle_mode --dataset_name bloodmnist --device cuda --create_kaggle_dataset
```

## Training Commands

### Individual Models
```bash
# Detector (Autoencoder)
python src/run.py --mode train --kaggle_mode --model_type detector --dataset_name bloodmnist --attack_type fgsm --attack_strength medium --epochs 50 --lr 0.001

# Reformer (Hypernet)
python src/run.py --mode train --kaggle_mode --model_type reformer --dataset_name bloodmnist --attack_type fgsm --attack_strength medium --epochs 50 --lr 0.001

# Classifier (ResNet18)
python src/run.py --mode train --kaggle_mode --model_type classifier --dataset_name bloodmnist --attack_type fgsm --attack_strength medium --epochs 50 --lr 0.001
```

### Pipeline Training
```bash
# Train all components
python src/run.py --mode train --kaggle_mode --model_type all --dataset_name bloodmnist --attack_type fgsm --attack_strength medium --epochs 50 --lr 0.001
```

### Different Attack Types
```bash
# FGSM attack
python src/run.py --mode train --kaggle_mode --model_type all --dataset_name bloodmnist --attack_type fgsm --attack_strength medium

# PGD attack
python src/run.py --mode train --kaggle_mode --model_type all --dataset_name bloodmnist --attack_type pgd --attack_strength medium

# Carlini attack
python src/run.py --mode train --kaggle_mode --model_type all --dataset_name bloodmnist --attack_type carlini --attack_strength medium
```

## Testing Commands

### Individual Model Testing
```bash
# Test detector
python src/run.py --mode test --kaggle_mode --test_type individual --model_type detector --dataset_name bloodmnist --attack_type fgsm --attack_strength medium

# Test reformer
python src/run.py --mode test --kaggle_mode --test_type individual --model_type reformer --dataset_name bloodmnist --attack_type fgsm --attack_strength medium

# Test classifier
python src/run.py --mode test --kaggle_mode --test_type individual --model_type classifier --dataset_name bloodmnist --attack_type fgsm --attack_strength medium
```

### Pipeline Testing
```bash
# Test all pipeline configurations
python src/run.py --mode test --kaggle_mode --test_type pipeline --dataset_name bloodmnist --attack_type fgsm --attack_strength medium
```

## Available Options

### Datasets
- `bloodmnist` (8 classes)
- `pathmnist` (9 classes)
- `dermamnist` (7 classes)
- `octmnist` (4 classes)
- `pneumoniamnist` (2 classes)
- `retinamnist` (5 classes)
- `breastmnist` (2 classes)
- `tissuemnist` (8 classes)

### Attack Types
- `fgsm` - Fast Gradient Sign Method
- `pgd` - Projected Gradient Descent
- `carlini` - Carlini & Wagner L2 attack

### Attack Strengths
- `weak` - Low perturbation
- `medium` - Moderate perturbation
- `strong` - High perturbation

### Model Types
- `detector` - Autoencoder-based detector
- `reformer` - Image reformer (hypernet/autoencoder/denoising)
- `classifier` - ResNet18 classifier
- `all` - Complete pipeline

## Command Structure

```bash
python src/run.py --mode [prepare_data|train|test] --kaggle_mode --dataset_name [dataset] --attack_type [attack] --attack_strength [strength] [additional_options]
```

## Output Files

### Data Generation
- Creates: `/kaggle/working/fl_purification_complete_dataset.zip`
- Upload this zip file to Kaggle as a dataset

### Training
- Saves models to: `./models/`
- `best_detector.pth` - Trained detector
- `best_reformer.pth` - Trained reformer
- `best_classifier.pth` - Trained classifier

### Testing
- Prints metrics to console
- Detector: F1, Precision, Recall, AUC, Filter rates
- Reformer: PSNR, SSIM, LPIPS
- Classifier: F1, Precision, Recall, AUC

## Complete Workflow Example

```bash
# 1. Generate data
python src/run.py --mode prepare_data --kaggle_mode --dataset_name bloodmnist --create_kaggle_dataset

# 2. Train pipeline
python src/run.py --mode train --kaggle_mode --model_type all --dataset_name bloodmnist --attack_type fgsm --attack_strength medium --epochs 50

# 3. Test pipeline
python src/run.py --mode test --kaggle_mode --test_type pipeline --dataset_name bloodmnist --attack_type fgsm --attack_strength medium
```

## Notes

- Use `--kaggle_mode` for Kaggle environment
- Remove `--kaggle_mode` for local environment
- Models are automatically saved during training
- Test results show filtering statistics for detector
- Pipeline testing compares 3 configurations: detector→reformer→classifier, detector→classifier, classifier only

