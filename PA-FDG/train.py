#!/usr/bin/env python3
"""
PA-FDG: Patient-Adaptive Feature Dependency Graph
Main training script for LOSO evaluation on CHB-MIT dataset.

Usage:
    python train.py --data_dir data/chbmit_features --gpu 0
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cp_protonet_loso_amlp_v3 import cp_protonet_loso_v7, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='PA-FDG Training')
    parser.add_argument('--data_dir', type=str, default='data/chbmit_features',
                        help='Path to feature directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pretrain_epochs', type=int, default=150,
                        help='Pretraining epochs')
    parser.add_argument('--adapt_epochs', type=int, default=80,
                        help='Adaptation epochs')
    parser.add_argument('--n_prototypes', type=int, default=10,
                        help='Number of prototypes per class')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Fusion weight for hybrid graph')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set seed
    set_seed(args.seed)
    
    # Patient list (16 patients used in paper, excluding chb06 and chb14)
    patients = [
        'chb01', 'chb02', 'chb03', 'chb05', 'chb07', 'chb08', 'chb09',
        'chb10', 'chb15', 'chb17', 'chb18', 'chb19', 'chb20', 'chb21',
        'chb22', 'chb23'
    ]
    
    print("=" * 70)
    print("PA-FDG: Patient-Adaptive Feature Dependency Graph")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Device: {device} (GPU {args.gpu})")
    print(f"  - Pretrain epochs: {args.pretrain_epochs}")
    print(f"  - Adapt epochs: {args.adapt_epochs}")
    print(f"  - Prototypes per class: {args.n_prototypes}")
    print(f"  - Fusion alpha: {args.alpha}")
    print(f"  - Patients: {len(patients)}")
    print("=" * 70)
    
    # Configure model
    cp_protonet_loso_v7._use_hybrid_graph = True
    cp_protonet_loso_v7._fusion_mode = 'fixed'
    cp_protonet_loso_v7._fixed_alpha = args.alpha
    cp_protonet_loso_v7._use_balanced_optimal = True
    cp_protonet_loso_v7._class_weight = 2.5
    
    # LOSO evaluation
    all_results = []
    
    for i, test_patient in enumerate(patients):
        train_patients = [p for p in patients if p != test_patient]
        
        print(f"\n{'=' * 70}")
        print(f"[{i+1}/{len(patients)}] Test patient: {test_patient}")
        print(f"{'=' * 70}")
        
        try:
            result = cp_protonet_loso_v7(
                test_patient=test_patient,
                train_patients=train_patients,
                num_adapt_samples=None,
                pretrain_epochs=args.pretrain_epochs,
                adapt_epochs=args.adapt_epochs,
                n_prototypes=args.n_prototypes,
                device=device
            )
            
            if result is not None:
                auc = result.get('AUC', result.get('auc', 0))
                sens = result.get('Sensitivity', result.get('sensitivity', 0))
                spec = result.get('Specificity', result.get('specificity', 0))
                
                all_results.append({
                    'patient': test_patient,
                    'auc': auc,
                    'sensitivity': sens,
                    'specificity': spec
                })
                
                print(f"\n{test_patient} Results:")
                print(f"  AUC: {auc:.2f}%")
                print(f"  Sensitivity: {sens:.2f}%")
                print(f"  Specificity: {spec:.2f}%")
                
        except Exception as e:
            print(f"Error on {test_patient}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    if all_results:
        avg_auc = np.mean([r['auc'] for r in all_results])
        avg_sens = np.mean([r['sensitivity'] for r in all_results])
        avg_spec = np.mean([r['specificity'] for r in all_results])
        
        print(f"\n{'Patient':<10} {'AUC':<10} {'Sens':<10} {'Spec':<10}")
        print("-" * 40)
        for r in all_results:
            print(f"{r['patient']:<10} {r['auc']:<10.2f} {r['sensitivity']:<10.2f} {r['specificity']:<10.2f}")
        print("-" * 40)
        print(f"{'Average':<10} {avg_auc:<10.2f} {avg_sens:<10.2f} {avg_spec:<10.2f}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
