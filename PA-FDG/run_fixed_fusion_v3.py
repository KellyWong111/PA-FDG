#!/usr/bin/env python3
"""
复现 Fixed Fusion (α=0.6) 结果 - v3 优化版
使用 GPU 0
"""

import os
import sys
import numpy as np
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用 GPU 0

# 导入 v3 版本
from cp_protonet_loso_amlp_v3 import cp_protonet_loso_v7, set_seed

set_seed(42)

print("=" * 80)
print("🔥 复现 Fixed Fusion (α=0.6) - v3 优化版 (GPU 0)")
print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

patients = [
    'chb01', 'chb02', 'chb03', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09',
    'chb10', 'chb14', 'chb15', 'chb17', 'chb18', 'chb19', 'chb20', 'chb21',
    'chb22', 'chb23'
]

target_results = {
    'chb01': 99.12, 'chb02': 98.73, 'chb03': 92.74, 'chb05': 85.78,
    'chb06': 71.47, 'chb07': 95.23, 'chb08': 94.96, 'chb09': 92.73,
    'chb10': 83.13, 'chb14': 70.85, 'chb15': 79.07, 'chb17': 97.64,
    'chb18': 97.25, 'chb19': 99.66, 'chb20': 99.52, 'chb21': 86.82,
    'chb22': 91.36, 'chb23': 97.45
}

print("\n配置:")
print("  - fusion_mode: fixed")
print("  - fixed_alpha: 0.6")
print("  - use_balanced_optimal: True")
print("  - class_weight: 2.5")
print("  - pretrain_epochs: 150")
print("  - adapt_epochs: 80")
print("  - n_prototypes: 10")
print("  - GPU: 0")
print("=" * 80)

cp_protonet_loso_v7._use_hybrid_graph = True
cp_protonet_loso_v7._fusion_mode = 'fixed'
cp_protonet_loso_v7._fixed_alpha = 0.6
cp_protonet_loso_v7._use_balanced_optimal = True
cp_protonet_loso_v7._class_weight = 2.5

all_results = []
for i, test_patient in enumerate(patients):
    train_patients = [p for p in patients if p != test_patient]
    
    print(f"\n{'=' * 80}")
    print(f"🚀 [{i+1}/18] 测试患者: {test_patient}")
    print(f"目标 AUC: {target_results[test_patient]:.2f}%")
    print(f"{'=' * 80}")
    
    try:
        result = cp_protonet_loso_v7(
            test_patient=test_patient,
            train_patients=train_patients,
            num_adapt_samples=None,
            pretrain_epochs=150,
            adapt_epochs=80,
            n_prototypes=10,
            device='cuda'
        )
        
        if result is not None:
            auc = result.get('AUC', result.get('auc', 0))
            sens = result.get('Sensitivity', result.get('sensitivity', 0))
            spec = result.get('Specificity', result.get('specificity', 0))
            
            all_results.append({
                'patient': test_patient,
                'auc': auc,
                'sens': sens,
                'spec': spec
            })
            
            target = target_results[test_patient] / 100
            diff = (auc - target) * 100
            status = "✅" if abs(diff) < 2 else ("⚠️" if abs(diff) < 5 else "❌")
            
            print(f"\n{test_patient} 结果:")
            print(f"  AUC:  {auc*100:.2f}% (目标: {target_results[test_patient]:.2f}%, 差距: {diff:+.2f}%) {status}")
            print(f"  Sens: {sens*100:.2f}%")
            print(f"  Spec: {spec*100:.2f}%")
    except Exception as e:
        print(f"❌ {test_patient} 错误: {e}")
        import traceback
        traceback.print_exc()

# 汇总
print("\n" + "=" * 80)
print("📊 最终结果汇总")
print("=" * 80)

if all_results:
    print(f"\n{'患者':<8} {'AUC':<10} {'Sens':<10} {'Spec':<10} {'目标AUC':<10} {'状态'}")
    print("-" * 70)
    
    for r in all_results:
        target = target_results[r['patient']]
        diff = r['auc'] * 100 - target
        status = "✅" if abs(diff) < 2 else ("⚠️" if abs(diff) < 5 else "❌")
        print(f"{r['patient']:<8} {r['auc']*100:>6.2f}%   {r['sens']*100:>6.2f}%   {r['spec']*100:>6.2f}%   {target:>6.2f}%    {status}")
    
    avg_auc = np.mean([r['auc'] for r in all_results])
    avg_sens = np.mean([r['sens'] for r in all_results])
    avg_spec = np.mean([r['spec'] for r in all_results])
    
    print("-" * 70)
    print(f"{'平均':<8} {avg_auc*100:>6.2f}%   {avg_sens*100:>6.2f}%   {avg_spec*100:>6.2f}%   {'90.75%':>8}")

print(f"\n⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
