# run.py

"""
Main entry point for FL-Purification
Handles all argument parsing and calls appropriate functions
"""

import argparse
import sys
import os
import wandb
from train import main as train_main
from test import main as test_main
from dataloader import (
    setup_kaggle_environment, 
    prepare_dataset_comprehensive, 
    create_kaggle_dataset_zip
)


def main():
    parser = argparse.ArgumentParser(description='FL-Purification: Main entry point')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'test', 'prepare_data'],
                       help='Mode to run: train, test, or prepare_data')
    
    # Model selection (for training)
    parser.add_argument('--model_type', type=str, default='all',
                       choices=['detector', 'reformer', 'classifier', 'all'],
                       help='Type of model to train (or all for pipeline)')
    
    # Test selection (for testing)
    parser.add_argument('--test_type', type=str, default='pipeline',
                       choices=['individual', 'pipeline'],
                       help='Type of testing to perform')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Dataset and attack configuration
    parser.add_argument('--dataset_name', type=str, default='bloodmnist',
                       choices = [
                                'bloodmnist', 'pathmnist', 'dermamnist', 'octmnist',
                                'pneumoniamnist', 'retinamnist', 'breastmnist', 'tissuemnist',
                                'organamnist', 'organcmnist', 'organsmnist', 'chestmnist', 'mnist', 'cifar10'],
                       help='MedMNIST dataset to use')
    
    parser.add_argument('--attack_type', type=str, default='carlini',
                       choices=['none', 'fgsm', 'pgd', 'carlini'],
                       help='Type of adversarial attack to use')
    parser.add_argument('--adv_ratio', type=float, default=0.3,
                       help='Ratio of adversarial examples in training data')
    
    # NEW: Clean training option
    parser.add_argument('--train_clean_only', action='store_true',
                       help='Train on clean data only (no adversarial examples in training)')
    
    # Reformer specific
    parser.add_argument('--reformer_type', type=str, default='hypernet',
                       choices=['hypernet', 'autoencoder', 'denoising_autoencoder'],
                       help='Type of reformer to use')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, default='./data_cache', help='Directory for prepared data')
    
    # Data preparation
    parser.add_argument('--attack_types', nargs='+', default=['fgsm', 'pgd', 'carlini'], help='Types of attacks to generate')
    parser.add_argument('--attack_strength', type=str, default='weak', 
                       choices=['weak', 'strong'], help='Attack strength for Kaggle data')
    parser.add_argument('--kaggle_mode', action='store_true', help='Enable Kaggle-compatible mode')
    parser.add_argument('--create_kaggle_dataset', action='store_true', help='Create Kaggle dataset zip file')
    
    # Pipeline behavior
    parser.add_argument('--reform_all', action='store_true',
                       help='Apply reformer to all images (not just detected adversarial ones)')
    
    # Logging
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    if args.mode == 'prepare_data':
        if args.kaggle_mode:
            print("Preparing comprehensive Kaggle dataset...")
            setup_kaggle_environment()
            prepare_dataset_comprehensive(args.dataset_name, args.device)
            
            if args.create_kaggle_dataset:
                create_kaggle_dataset_zip()
                print("Kaggle dataset zip file created!")
        else:
            print("Preparing dataset and generating attacks...")
            print("Local data preparation is not supported in this version.")
            print("Please use --kaggle_mode for data preparation.")
            print("Data preparation completed!")
            return
    
    # Common configuration
    config = {
        'batch_size': args.batch_size,
        'device': args.device,
        'wandb': args.wandb,
        'dataset_name': args.dataset_name,
        'attack_type': args.attack_type,
        'adv_ratio': args.adv_ratio,
        'reformer_type': args.reformer_type,
        'data_dir': args.data_dir,
        'reform_all': args.reform_all,
        'kaggle_mode': args.kaggle_mode,
        'attack_strength': args.attack_strength,
        'train_clean_only': args.train_clean_only  # NEW: Add clean training flag
    }
    
    # Initialize wandb if requested
    if args.wandb:
        os.environ["WANDB_API_KEY"] = "93e0092bafec12515dce3493023285e27311c27a"
        wandb.login(key=os.getenv("WANDB_API_KEY", default="93e0092bafec12515dce3493023285e27311c27a"))
        
        if args.mode == 'train':
            wandb.init(
                project="fl-purification",
                entity="invi-bhagyesh-manipal",
                config={
                    "Model Type": args.model_type,
                    "Epochs": args.epochs,
                    "Batch Size": args.batch_size,
                    "Learning Rate": args.lr,
                    "Dataset": args.dataset_name,
                    "Attack Type": args.attack_type,
                    "Adversarial Ratio": args.adv_ratio,
                    "Reformer Type": args.reformer_type,
                    "Train Clean Only": args.train_clean_only  # NEW: Log clean training flag
                }
            )
        else:  # test mode
            wandb.init(
                project="fl-purification",
                entity="invi-bhagyesh-manipal",
                config={
                    "Test Type": args.test_type,
                    "Model Type": args.model_type,
                    "Batch Size": args.batch_size,
                    "Dataset": args.dataset_name,
                    "Attack Type": args.attack_type,
                    "Adversarial Ratio": args.adv_ratio,
                    "Reformer Type": args.reformer_type
                }
            )
    
    if args.mode == 'train':
        # Add training-specific config
        config.update({
            'epochs': args.epochs,
            'lr': args.lr,
            'pipeline_type': args.model_type
        })
        
        # Run training
        train_main(config)
    
    elif args.mode == 'test':
        # Add testing-specific config
        config.update({
            'test_type': args.test_type,
            'model_type': args.model_type
        })
        
        # Run testing
        test_main(config, args.test_type, args.model_type)

if __name__ == '__main__':
    main()