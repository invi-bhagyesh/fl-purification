# test.py
"""
Comprehensive testing script for FL-Purification
Tests individual models and pipeline configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips

# Import models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from models.DAE import DenoisingAutoEncoder
from models.hypernet import AdaptiveLaplacianPyramidUNet
from models.resnet18 import ResNet18_MedMNIST
from models.victim import SimpleCNN
from models.AE import SimpleAutoencoder

# Import utilities
from dataloader import (
    load_kaggle_dataset, 
    create_dataloader_from_kaggle_data,
    load_multiple_attacks,
    get_dataset_info,
    list_available_kaggle_datasets
)
from utils.Attacks import fgsm_attack, pgd_attack, carlini_attack
from utils.utils import batch_psnr_ssim

class Detector(nn.Module):
    """Autoencoder-based detector using reconstruction error"""
    def __init__(self, input_channels=3):
        super(Detector, self).__init__()
        # Use autoencoder as detector - reconstruction error indicates adversarial
        self.autoencoder = SimpleAutoencoder()
        
        # Threshold for reconstruction error (can be tuned)
        self.threshold = 0.1  # Images with reconstruction error > threshold are considered adversarial
    
    def forward(self, x):
        # Get reconstruction from autoencoder
        reconstruction = self.autoencoder(x)
        
        # Calculate reconstruction error (MSE between input and reconstruction)
        reconstruction_error = F.mse_loss(x, reconstruction, reduction='none').mean(dim=[1, 2, 3])
        
        # Create binary classification based on threshold
        # 0 = clean (low reconstruction error), 1 = adversarial (high reconstruction error)
        is_adversarial = (reconstruction_error > self.threshold).float()
        
        # Convert to 2-class output format [clean_prob, adversarial_prob]
        clean_prob = 1.0 - is_adversarial
        adversarial_prob = is_adversarial
        
        # Stack into 2-class output
        output = torch.stack([clean_prob, adversarial_prob], dim=1)
        
        return output

class Pipeline(nn.Module):
    """Flexible pipeline that can combine detector, reformer, and classifier"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # Initialize components based on pipeline type
        self.detector = None
        self.reformer = None
        self.classifier = None
        
        if 'detector' in config['pipeline_type']:
            self.detector = Detector().to(self.device)
        
        if 'reformer' in config['pipeline_type']:
            reformer_type = config.get('reformer_type', 'hypernet')
            if reformer_type == 'hypernet':
                self.reformer = AdaptiveLaplacianPyramidUNet(
                    encoder_name='resnet34',
                    encoder_weights='imagenet',
                    decoder_channels=(256, 128, 64, 32, 16),
                    num_pyramid_levels=5,
                    kernel_size=5
                ).to(self.device)
            elif reformer_type == 'autoencoder':
                self.reformer = SimpleAutoencoder().to(self.device)
            elif reformer_type == 'denoising_autoencoder':
                self.reformer = DenoisingAutoEncoder(
                    image_shape=(3, 28, 28), 
                    structure=[64, 128, 256, "max", 128, 64]
                ).to(self.device)
        
        if 'classifier' in config['pipeline_type']:
            if config['dataset_name'] == 'bloodmnist':
                num_classes = 8
            elif config['dataset_name'] == 'pathmnist':
                num_classes = 9
            elif config['dataset_name'] == 'dermamnist':
                num_classes = 7
            elif config['dataset_name'] == 'octmnist':
                num_classes = 4
            elif config['dataset_name'] == 'pneumoniamnist':
                num_classes = 2
            elif config['dataset_name'] == 'retinamnist':
                num_classes = 5
            elif config['dataset_name'] == 'breastmnist':
                num_classes = 2
            elif config['dataset_name'] == 'tissuemnist':
                num_classes = 8
            elif config['dataset_name'] == 'organamnist':
                num_classes = 11
            elif config['dataset_name'] == 'organcmnist':
                num_classes = 11
            elif config['dataset_name'] == 'organsmnist':
                num_classes = 11
            elif config['dataset_name'] == 'chestmnist':
                num_classes = 14
            elif config['dataset_name'] == 'mnist':
                num_classes = 10
            elif config['dataset_name'] == 'cifar10':
                num_classes = 10
            else:
                num_classes = 8  # default
            
            # use ResNet as classifier
            self.classifier = ResNet18_MedMNIST(num_classes=num_classes).to(self.device)
    
    def forward(self, x):
        """Forward pass through the pipeline"""
        if self.detector is not None:
            detection = self.detector(x)
            # Only pass clean images (class 0), discard adversarial (class 1)
            clean_mask = (detection.argmax(dim=1) == 0)  # 0 = clean, 1 = adversarial
            
            if clean_mask.any():
                # Only process clean images
                clean_images = x[clean_mask]
                
                if self.reformer is not None:
                    clean_images = self.reformer(clean_images)[0]
                
                if self.classifier is not None:
                    clean_outputs = self.classifier(clean_images)
                    
                    # Create output tensor for all images
                    batch_size = x.size(0)
                    num_classes = clean_outputs.size(1)
                    all_outputs = torch.zeros(batch_size, num_classes, device=x.device)
                    
                    # Fill in outputs for clean images only
                    all_outputs[clean_mask] = clean_outputs
                    
                    return all_outputs
                else:
                    return clean_images
            else:
                # No clean images, return zeros
                if self.classifier is not None:
                    batch_size = x.size(0)
                    num_classes = self.classifier(torch.zeros(1, *x.shape[1:], device=x.device)).size(1)
                    return torch.zeros(batch_size, num_classes, device=x.device)
                else:
                    return torch.zeros_like(x)
        
        elif self.reformer is not None:
            # Only reformer (no detector)
            x = self.reformer(x)[0]
            
            if self.classifier is not None:
                output = self.classifier(x)
                return output
        
        elif self.classifier is not None:
            # Only classifier (no detector or reformer)
            output = self.classifier(x)
            return output
        
        return x

def get_num_classes(dataset_name):
    """Get number of classes for a given MedMNIST dataset"""
    if dataset_name == 'bloodmnist': # doing - invi
        return 8
    elif dataset_name == 'pathmnist': # doing - akshat
        return 9
    elif dataset_name == 'dermamnist': # doing - akshat
        return 7
    elif dataset_name == 'octmnist': # doing - invi *
        return 4
    elif dataset_name == 'pneumoniamnist': # doing - invi *
        return 2
    elif dataset_name == 'retinamnist': 
        return 5
    elif dataset_name == 'breastmnist':
        return 2
    elif dataset_name == 'tissuemnist':
        return 8
    elif dataset_name == 'organamnist': # doing - dishita *
        return 11
    elif dataset_name == 'organcmnist':
        return 11
    elif dataset_name == 'organsmnist':
        return 11
    elif dataset_name == 'chestmnist': 
        return 14
    elif dataset_name == 'mnist':
        return 10
    elif dataset_name == 'cifar10':
        return 10
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def test_individual_model(config, model_type):
    """Test individual model"""
    print(f"Testing {model_type} model...")
    
    # Check if we're in Kaggle mode
    kaggle_mode = config.get('kaggle_mode', False)
    
    if kaggle_mode:
        print("Using Kaggle dataloader for testing...")
        
        # Check available datasets
        list_available_kaggle_datasets()
        
        # Get dataset info
        dataset_info = get_dataset_info(config['dataset_name'])
        if not dataset_info:
            print(f"No Kaggle dataset found for {config['dataset_name']}")
            return None
        
        # Load test data from Kaggle dataset
        attack_type = config['attack_type']
        strength = config.get('attack_strength', 'strong')
        
        if attack_type == 'none':
            print("No attack specified, using clean data only")
            attack_type = 'fgsm'  # dummy, will be filtered out
        
        try:
            test_data = load_kaggle_dataset(
                config['dataset_name'], 
                attack_type, 
                strength, 
                split='test'
            )
            
            test_loader = create_dataloader_from_kaggle_data(
                test_data, 
                batch_size=config['batch_size'], 
                shuffle=False
            )
            
            print(f"Successfully loaded Kaggle test data: {len(test_loader)} batches")
            
        except Exception as e:
            print(f"Error loading Kaggle test data: {e}")
            return None
    
    else:
        # Legacy mode - use local data preparation
        print("Using local data preparation for testing...")
        
        # Load prepared data
        clean_data = load_prepared_data(config['dataset_name'], split='test', data_dir=config.get('data_dir', './data_cache'))
        adv_data = load_prepared_data(config['dataset_name'], config['attack_type'], split='test', data_dir=config.get('data_dir', './data_cache'))
        
        # Create combined test dataset
        test_images, test_pert_labels, test_true_labels = create_combined_dataset(
            clean_data, adv_data, config.get('adv_ratio', 0.3)
        )
        
        test_dataset = TensorDataset(test_images, test_pert_labels, test_true_labels)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    if model_type == 'detector':
        model = Detector().to(device)
        model_path = './models/best_detector.pth'
    elif model_type == 'reformer':
        reformer_type = config.get('reformer_type', 'hypernet')
        if reformer_type == 'hypernet':
            model = AdaptiveLaplacianPyramidUNet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                decoder_channels=(256, 128, 64, 32, 16),
                num_pyramid_levels=5,
                kernel_size=5
            ).to(device)
        elif reformer_type == 'autoencoder':
            model = SimpleAutoencoder().to(device)
        elif reformer_type == 'denoising_autoencoder':
            model = DenoisingAutoEncoder(
                image_shape=(3, 28, 28), 
                structure=[64, 128, 256, "max", 128, 64]
            ).to(device)
        model_path = './models/best_reformer.pth'
    elif model_type == 'classifier':
        num_classes = get_num_classes(config['dataset_name'])
        model = ResNet18_MedMNIST(num_classes=num_classes).to(device)
        model_path = './models/best_classifier.pth'
    
    # Load trained model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded {model_type} model from {model_path}")
    else:
        print(f"Model not found at {model_path}")
        return None
    
    # Test model
    model.eval()
    
    # Initialize LPIPS for perceptual similarity
    try:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    except:
        print("LPIPS not available, skipping perceptual similarity metric")
        lpips_fn = None
    
    # Metrics storage
    all_preds = []
    all_labels = []
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, pert_labels, true_labels in tqdm(test_loader, desc=f'Testing {model_type}'):
            images = images.to(device)
            pert_labels = pert_labels.long().to(device)
            true_labels = true_labels.long().to(device)
            
            # Forward pass
            if model_type == 'reformer':
                if hasattr(model, 'forward') and model.forward.__code__.co_argcount > 1:
                    outputs, _ = model(images)
                else:
                    outputs = model(images)
                
                # Calculate image quality metrics
                psnr, ssim = batch_psnr_ssim(images, outputs)
                total_psnr += psnr
                total_ssim += ssim
                
                if lpips_fn is not None:
                    lpips_val = lpips_fn(images, outputs).mean().item()
                    total_lpips += lpips_val
                
                num_batches += 1
            else:
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                
                if model_type == 'detector':
                    all_labels.extend(pert_labels.cpu().numpy())
                else:  # classifier
                    all_labels.extend(true_labels.cpu().numpy())
    
    # Calculate and print metrics
    if model_type == 'reformer':
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        avg_lpips = total_lpips / num_batches if lpips_fn is not None else 0
        
        print(f"\n{model_type.upper()} Results:")
        print(f"PSNR: {avg_psnr:.2f}")
        print(f"SSIM: {avg_ssim:.4f}")
        if lpips_fn is not None:
            print(f"LPIPS: {avg_lpips:.4f}")
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips
        }
    else:
        # Calculate classification metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        try:
            auc = roc_auc_score(all_labels, all_preds, average='weighted')
        except:
            auc = 0.0
        
        print(f"\n{model_type.upper()} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

def test_pipeline_configurations(config):
    """Test the three pipeline configurations:
    1. detector->reformer->base classifier
    2. detector->base classifier  
    3. base classifier (base classifier=resnet18)
    """
    print("Testing pipeline configurations...")
    
    # Check if we're in Kaggle mode
    kaggle_mode = config.get('kaggle_mode', False)
    
    if kaggle_mode:
        print("Using Kaggle dataloader for testing...")
        
        # Check available datasets
        list_available_kaggle_datasets()
        
        # Get dataset info
        dataset_info = get_dataset_info(config['dataset_name'])
        if not dataset_info:
            print(f"No Kaggle dataset found for {config['dataset_name']}")
            return None
        
        # Load test data from Kaggle dataset
        attack_type = config['attack_type']
        strength = config.get('attack_strength', 'strong')
        
        if attack_type == 'none':
            print("No attack specified, using clean data only")
            attack_type = 'fgsm' 
        
        try:
            test_data = load_kaggle_dataset(
                config['dataset_name'], 
                attack_type, 
                strength, 
                split='test'
            )
            
            test_loader = create_dataloader_from_kaggle_data(
                test_data, 
                batch_size=config['batch_size'], 
                shuffle=False
            )
            
            print(f"Successfully loaded Kaggle test data: {len(test_loader)} batches")
            
        except Exception as e:
            print(f"Error loading Kaggle test data: {e}")
            return None
    
    else:
        # Legacy mode - use local data preparation
        print("Using local data preparation for testing...")
        
        # Load prepared data
        clean_data = load_prepared_data(config['dataset_name'], split='test', data_dir=config.get('data_dir', './data_cache'))
        adv_data = load_prepared_data(config['dataset_name'], config['attack_type'], split='test', data_dir=config.get('data_dir', './data_cache'))
        
        # Create combined test dataset
        test_images, test_pert_labels, test_true_labels = create_combined_dataset(
            clean_data, adv_data, config.get('adv_ratio', 0.3)
        )
        
        test_dataset = TensorDataset(test_images, test_pert_labels, test_true_labels)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Configuration 1: detector->reformer->base classifier
    print("\n=== Testing Configuration 1: detector->reformer->base classifier ===")
    config1 = config.copy()
    config1['pipeline_type'] = 'detector+reformer+classifier'
    pipeline1 = Pipeline(config1)
    
    # Load trained models
    if os.path.exists('./models/best_detector.pth'):
        pipeline1.detector.load_state_dict(torch.load('./models/best_detector.pth'))
    if os.path.exists('./models/best_reformer.pth'):
        pipeline1.reformer.load_state_dict(torch.load('./models/best_reformer.pth'))
    if os.path.exists('./models/best_classifier.pth'):
        pipeline1.classifier.load_state_dict(torch.load('./models/best_classifier.pth'))
    
    results1 = test_pipeline_with_metrics(pipeline1, test_loader, config1)
    
    # Configuration 2: detector->base classifier
    print("\n=== Testing Configuration 2: detector->base classifier ===")
    config2 = config.copy()
    config2['pipeline_type'] = 'detector+classifier'
    pipeline2 = Pipeline(config2)
    
    # Load trained models
    if os.path.exists('./models/best_detector.pth'):
        pipeline2.detector.load_state_dict(torch.load('./models/best_detector.pth'))
    if os.path.exists('./models/best_classifier.pth'):
        pipeline2.classifier.load_state_dict(torch.load('./models/best_classifier.pth'))
    
    results2 = test_pipeline_with_metrics(pipeline2, test_loader, config2)
    
    # Configuration 3: base classifier only
    print("\n=== Testing Configuration 3: base classifier only ===")
    config3 = config.copy()
    config3['pipeline_type'] = 'classifier'
    pipeline3 = Pipeline(config3)
    
    # Load trained model
    if os.path.exists('./models/best_classifier.pth'):
        pipeline3.classifier.load_state_dict(torch.load('./models/best_classifier.pth'))
    
    results3 = test_pipeline_with_metrics(pipeline3, test_loader, config3)
    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE CONFIGURATION COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'Detector F1':<12} {'Reformer PSNR':<15} {'Classifier F1':<15}")
    print("-"*80)
    print(f"{'1. Det->Ref->Cls':<30} {results1.get('detector_f1', 'N/A'):<12.4f} {results1.get('reformer_psnr', 'N/A'):<15.2f} {results1.get('classifier_f1', 'N/A'):<15.4f}")
    print(f"{'2. Det->Cls':<30} {results2.get('detector_f1', 'N/A'):<12.4f} {'N/A':<15} {results2.get('classifier_f1', 'N/A'):<15.4f}")
    print(f"{'3. Cls Only':<30} {'N/A':<12} {'N/A':<15} {results3.get('classifier_f1', 'N/A'):<15.4f}")
    print("="*80)
    
    return results1, results2, results3

def test_pipeline_with_metrics(pipeline, test_loader, config):
    """Test pipeline with comprehensive metrics"""
    device = pipeline.device
    pipeline.eval()
    
    # Initialize LPIPS for perceptual similarity
    try:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    except:
        print("LPIPS not available, skipping perceptual similarity metric")
        lpips_fn = None
    
    # Metrics storage
    detector_metrics = {'predictions': [], 'labels': []}
    reformer_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    classifier_metrics = {'predictions': [], 'labels': []}
    
    with torch.no_grad():
        for images, pert_labels, true_labels in tqdm(test_loader, desc='Testing Pipeline'):
            images = images.to(device)
            pert_labels = pert_labels.long().to(device)
            true_labels = true_labels.long().to(device)
            
            # Test detector
            if pipeline.detector is not None:
                detector_output = pipeline.detector(images)
                _, predicted = detector_output.max(1)
                detector_metrics['predictions'].extend(predicted.cpu().numpy())
                detector_metrics['labels'].extend(pert_labels.cpu().numpy())
            
            # Test reformer
            if pipeline.reformer is not None:
                if hasattr(pipeline.reformer, 'forward') and pipeline.reformer.forward.__code__.co_argcount > 1:
                    reformer_output, _ = pipeline.reformer(images)
                else:
                    reformer_output = pipeline.reformer(images)
                
                # Calculate image quality metrics
                psnr, ssim = batch_psnr_ssim(images, reformer_output)
                reformer_metrics['psnr'].append(psnr)
                reformer_metrics['ssim'].append(ssim)
                
                if lpips_fn is not None:
                    lpips_val = lpips_fn(images, reformer_output).mean().item()
                    reformer_metrics['lpips'].append(lpips_val)
            
            # Test classifier
            if pipeline.classifier is not None:
                classifier_output = pipeline.forward(images)
                _, predicted = classifier_output.max(1)
                classifier_metrics['predictions'].extend(predicted.cpu().numpy())
                classifier_metrics['labels'].extend(true_labels.cpu().numpy())
    
    # Calculate final metrics
    results = {}
    
    # Detector metrics
    if pipeline.detector is not None and detector_metrics['predictions']:
        results['detector_f1'] = f1_score(detector_metrics['labels'], detector_metrics['predictions'], average='weighted')
        results['detector_precision'] = precision_score(detector_metrics['labels'], detector_metrics['predictions'], average='weighted')
        results['detector_recall'] = recall_score(detector_metrics['labels'], detector_metrics['predictions'], average='weighted')
        results['detector_auc'] = roc_auc_score(detector_metrics['labels'], detector_metrics['predictions'], average='weighted')
        print(f"Detector - F1: {results['detector_f1']:.4f}, Precision: {results['detector_precision']:.4f}, Recall: {results['detector_recall']:.4f}, AUC: {results['detector_auc']:.4f}")
    
    # Reformer metrics
    if pipeline.reformer is not None and reformer_metrics['psnr']:
        results['reformer_psnr'] = np.mean(reformer_metrics['psnr'])
        results['reformer_ssim'] = np.mean(reformer_metrics['ssim'])
        results['reformer_lpips'] = np.mean(reformer_metrics['lpips']) if reformer_metrics['lpips'] else 0
        print(f"Reformer - PSNR: {results['reformer_psnr']:.2f}, SSIM: {results['reformer_ssim']:.4f}, LPIPS: {results['reformer_lpips']:.4f}")
    
    # Classifier metrics
    if pipeline.classifier is not None and classifier_metrics['predictions']:
        results['classifier_f1'] = f1_score(classifier_metrics['labels'], classifier_metrics['predictions'], average='weighted')
        results['classifier_precision'] = precision_score(classifier_metrics['labels'], classifier_metrics['predictions'], average='weighted')
        results['classifier_recall'] = recall_score(classifier_metrics['labels'], classifier_metrics['predictions'], average='weighted')
        results['classifier_auc'] = roc_auc_score(classifier_metrics['labels'], classifier_metrics['predictions'], average='weighted')
        print(f"Classifier - F1: {results['classifier_f1']:.4f}, Precision: {results['classifier_precision']:.4f}, Recall: {results['classifier_recall']:.4f}, AUC: {results['classifier_auc']:.4f}")
    
    return results

def main(config, test_type='pipeline', model_type='detector'):
    """Main testing function that takes a config dictionary and test parameters"""
    print(f"Testing {test_type}...")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Attack type: {config['attack_type']}")
    print(f"Batch size: {config['batch_size']}")
    
    # Check if we're in Kaggle mode
    kaggle_mode = config.get('kaggle_mode', False)
    
    if kaggle_mode:
        print("Using Kaggle mode for testing...")
        # In Kaggle mode, we don't need to check local data directory
        # The dataloader functions will handle dataset availability
    else:
        # Check if data exists for local mode
        if not os.path.exists(os.path.join(config.get('data_dir', './data_cache'), config['dataset_name'])):
            print(f"Prepared data not found in {config.get('data_dir', './data_cache')}. Please run training with --prepare_data first.")
            return
    
    # Perform testing
    if test_type == 'individual':
        results = test_individual_model(config, model_type)
        if results:
            print(f"\n{model_type.upper()} testing completed!")
    else:  # pipeline
        results = test_pipeline_configurations(config)
        print("\nPipeline testing completed!") 