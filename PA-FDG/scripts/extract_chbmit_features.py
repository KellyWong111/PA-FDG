#!/usr/bin/env python3
"""
CHB-MIT数据集特征提取脚本
直接从pkl格式的LOSO数据集提取特征
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import yaml
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from optparse import OptionParser
import mne

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ==================== 特征提取类（从你的代码复制） ====================

def safe_log10(x, eps=1e-10):     
    result = np.where(x > eps, x, -10)     
    np.log10(result, out=result, where=result > 0)     
    return result

def relative_log_power(data, window=256, fs=256, overlap=0.,
                       frequencies=[[0.1, 4], [4, 8], [8, 12], [12, 30], [30, 70], [70, 180]]):
    from scipy.signal import welch
    noverlap = int(window * overlap)

    freqs, power = welch(data, fs=fs, nperseg=window, noverlap=noverlap)
    out = []
    if frequencies is None:
        out = power
    else:
        for fr in frequencies:
            tmp = (freqs >= fr[0]) & (freqs < fr[1])
            if np.any(tmp):
                output = (power[:, tmp].mean(1))
                out.append(output)
            else:
                out.append(np.zeros(power.shape[0]))
    out_arr = np.array(out)
    total_power = np.sum(out_arr, 0) + 1e-10
    return safe_log10(out_arr / total_power)

class Windower:
    """降采样+窗口化"""
    def __init__(self, window=5, overlap=0, srate=256):  # 改为5秒，匹配SGSTAN
        self.window = window  # 秒
        self.overlap = overlap
        self.srate = srate  # 目标采样率（降采样后）

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: (n_batch, n_channels, n_samples) 或 (n_channels, n_samples)
        输出: (n_windows, n_channels, window_samples) 保持原采样率
        """
        # 处理输入维度
        if X.ndim == 2:
            # (n_channels, n_samples) -> 添加batch维度
            X = X[np.newaxis, ...]
        
        # 对第一个batch进行窗口化（假设batch_size=1）
        X = X[0]  # (n_channels, n_samples)
        
        # 直接窗口化，不降采样
        wi = int(self.window * self.srate)
        ov = int(self.overlap * wi)
        
        out = []
        nSamples = X.shape[1]
        ind = list(range(0, nSamples - wi + 1, wi - ov))
        
        for idx in ind:
            sl = slice(idx, idx + wi)
            out.append(X[:, sl])
        
        return np.array(out) if out else np.array([]).reshape(0, X.shape[0], 0)

class RelativeLogPower:
    def __init__(self, window=256, overlap=0.0, fs=40, 
                 frequencies=[[0.1, 4], [4, 8], [8, 12], [12, 40]]):
        self.window = window
        self.overlap = overlap
        self.fs = fs
        self.frequencies = frequencies

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """X: (Nt, Ne, Ns)"""
        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            try:
                S = relative_log_power(X[i], window=self.window, fs=self.fs,
                                       overlap=self.overlap, frequencies=self.frequencies)
                out.append(S.T)
            except Exception as e:
                n_freqs = len(self.frequencies)
                out.append(np.zeros((Ne, n_freqs)))

        return np.array(out)

class Coherences:
    def __init__(self, window=256, overlap=0.5, fs=40,
                 frequencies=[[0.1, 4], [4, 8], [8, 12]], 
                 aggregate=False, return_global_median=False):
        self.window = window
        self.overlap = overlap
        self.fs = fs
        self.frequencies = frequencies
        self.aggregate = aggregate
        self.return_global_median = return_global_median

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy.signal import coherence
        
        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            try:
                pair_coherences = []
                
                # 计算所有通道对
                for ch1 in range(Ne):
                    for ch2 in range(ch1+1, Ne):
                        try:
                            freqs, coh = coherence(X[i, ch1], X[i, ch2], 
                                                 fs=self.fs, nperseg=self.window)
                            
                            for freq_band in self.frequencies:
                                mask = (freqs >= freq_band[0]) & (freqs < freq_band[1])
                                if np.any(mask):
                                    band_coh = np.mean(coh[mask])
                                else:
                                    band_coh = 0.0
                                pair_coherences.append(band_coh)
                        except:
                            for _ in self.frequencies:
                                pair_coherences.append(0.0)
                
                # 根据模式返回
                if self.return_global_median:
                    # 返回中位数（单一值）
                    median_coh = np.median(pair_coherences) if pair_coherences else 0.0
                    out.append(np.array([[median_coh]]))
                else:
                    # 返回所有通道对的相干性
                    out.append(np.array([pair_coherences]))
                
            except Exception as e:
                print(f"⚠️ Coherences计算错误: {e}")
                if self.return_global_median:
                    out.append(np.array([[0.0]]))
                else:
                    n_pairs = Ne * (Ne - 1) // 2
                    n_features = n_pairs * len(self.frequencies)
                    out.append(np.zeros((1, n_features)))

        return np.array(out)

class BasicStats:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import scipy as sp
        out = []
        for x in X:
            try:
                x_clean = np.where(np.isfinite(x), x, 0)
                
                m = np.mean(x_clean, 1)
                sd = np.std(x_clean, 1)
                ku = sp.stats.kurtosis(x_clean, 1, nan_policy='omit')
                sk = sp.stats.skew(x_clean, 1, nan_policy='omit')
                p90 = np.percentile(x_clean, 90, axis=1)
                p10 = np.percentile(x_clean, 10, axis=1)
                
                tmp = np.c_[m, sd, ku, sk, p90, p10]
                
                if not np.all(np.isfinite(tmp)):
                    tmp = np.where(np.isfinite(tmp), tmp, 0)
                    
                out.append(tmp)
            except Exception as e:
                print(f"⚠️ BasicStats错误: {e}")
                out.append(np.zeros((x.shape[0], 6)))

        return np.array(out)

class AsymmetryFeatures(BasicStats):
    # CHB-MIT的对称通道对（需要根据实际通道配置调整）
    LEFT_RIGHT_PAIRS = [
        (0, 4),   # FP1-F7 vs FP2-F8
        (1, 5),   # F7-T7 vs F8-T8
        (2, 6),   # T7-P7 vs T8-P8
        (3, 7),   # P7-O1 vs P8-O2
        (4, 8),   # FP1-F3 vs FP2-F4
        (5, 9),   # F3-C3 vs F4-C4
    ]

    def __init__(self, abs_flag=False):
        self.abs_flag = abs_flag

    def transform(self, X):
        X_stats = BasicStats.transform(self, X)
        out = []
        
        for x in X_stats:
            try:
                tmp = []
                for ldx, rdx in self.LEFT_RIGHT_PAIRS:
                    if ldx < len(x) and rdx < len(x):
                        left_vals = x[ldx]
                        right_vals = x[rdx]
                        
                        denominator = np.abs(left_vals) + np.abs(right_vals) + 1e-10
                        
                        if self.abs_flag:
                            asymm = np.abs(left_vals - right_vals) / denominator
                        else:
                            asymm = (left_vals - right_vals) / denominator
                        
                        asymm = np.where(np.isfinite(asymm), asymm, 0)
                        tmp.extend(asymm)
                    else:
                        tmp.extend([0.0] * 6)
                
                result = np.array(tmp)
                if np.all(result == 0):
                    result += 1e-6
                
                out.append(result)
            except Exception as e:
                print(f"⚠️ AsymmetryFeatures错误: {e}")
                expected_length = len(self.LEFT_RIGHT_PAIRS) * 6
                out.append(np.zeros(expected_length) + 1e-6)

        return np.array(out)

class AutoCorrMat:
    def __init__(self, order=6, subsample=4, eigenmode=True):
        self.order = order
        self.subsample = subsample
        self.eigenmode = eigenmode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for x in X:
            tmp = []
            for i, a in enumerate(x):
                try:
                    a_sub = a[::self.subsample]
                    if len(a_sub) < 10:
                        tmp.append(0.1)
                        continue
                    
                    autocorrs = []
                    for lag in range(1, min(self.order + 1, len(a_sub)//2)):
                        if len(a_sub) > lag:
                            corr = np.corrcoef(a_sub[:-lag], a_sub[lag:])[0, 1]
                            autocorrs.append(corr if np.isfinite(corr) else 0.0)
                        else:
                            autocorrs.append(0.0)
                    
                    if self.eigenmode:
                        max_corr = np.max(np.abs(autocorrs)) if autocorrs else 0.1
                        tmp.append(max_corr)
                    else:
                        corr_matrix = np.diag(autocorrs[:self.order])
                        tmp.append(corr_matrix)
                        
                except Exception as e:
                    tmp.append(0.1 if self.eigenmode else np.eye(self.order) * 0.1)
            
            out.append(tmp)
            
        if self.eigenmode:
            return np.array(out)[..., np.newaxis]
        else:
            return np.array(out).transpose(0,2,3,1)

class ARError:
    def __init__(self, order=5, subsample=4):
        self.order = order
        self.subsample = subsample

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy import linalg
        out = []
        
        for x in X:
            tmp = []
            for a in x:
                try:
                    a_sub = a[::self.subsample]
                    if len(a_sub) < self.order + 5:
                        tmp.append(np.array([0.1] * (self.order + 1)))
                        continue
                    
                    N = len(a_sub)
                    A = np.zeros((N - self.order, self.order))
                    b = a_sub[self.order:]
                    
                    for i in range(self.order):
                        A[:, i] = a_sub[self.order-1-i:N-1-i]
                    
                    try:
                        coeffs, residuals, rank, s = linalg.lstsq(A, b)
                        
                        if len(residuals) > 0 and rank == self.order:
                            mse = residuals[0] / (len(b) - self.order)
                            cov_matrix = mse * linalg.inv(A.T @ A + np.eye(self.order) * 1e-6)
                            bse = np.sqrt(np.diag(cov_matrix))
                            bse = np.concatenate([[mse], bse])
                        else:
                            bse = np.array([0.1] * (self.order + 1))
                    except:
                        bse = np.array([0.1] * (self.order + 1))
                    
                    tmp.append(bse)
                    
                except Exception as e:
                    tmp.append(np.array([0.1] * (self.order + 1)))
                    
            out.append(tmp)
        return np.array(out)

class rqa_channel:
    def __init__(self, hp=0.5, lp=4, tau=2, emb_dim=3, sfreq=40):
        self.hp = hp
        self.lp = lp
        self.tau = tau
        self.emb_dim = emb_dim
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = np.zeros([X.shape[0], X.shape[1], 7])
        
        for cur, x in enumerate(X):
            for idx, ch in enumerate(x):
                try:
                    ch_clean = ch[np.isfinite(ch)]
                    if len(ch_clean) < 100:
                        out[cur, idx] = [0.1] * 7
                        continue
                    
                    rr = np.std(ch_clean) / (np.mean(np.abs(ch_clean)) + 1e-10)
                    det = np.mean(np.abs(np.diff(ch_clean)))
                    lam = np.var(ch_clean)
                    l_max = np.max(np.abs(ch_clean))
                    
                    hist, _ = np.histogram(ch_clean, bins=20)
                    hist = hist / (np.sum(hist) + 1e-10)
                    l_entr = -np.sum(hist * np.log(hist + 1e-10))
                    
                    l_mean = np.mean(ch_clean)
                    tt = np.median(np.abs(ch_clean))
                    
                    features = [rr, det, lam, l_max, l_entr, l_mean, tt]
                    features = [f if np.isfinite(f) else 0.1 for f in features]
                    
                    out[cur, idx] = features
                    
                except Exception as e:
                    out[cur, idx] = [0.1] * 7
                    
        return out

# ==================== 流水线构建 ====================

def build_feature_pipeline(feature_config, srate=256):
    """根据配置构建特征提取流水线"""
    from sklearn.pipeline import Pipeline
    
    steps = []
    for step_dict in feature_config['preproc']:
        for method_name, params in step_dict.items():
            # 修正参数中的srate
            if 'srate' in params:
                params['srate'] = srate
            
            # 创建特征提取器
            if method_name == 'Windower':
                extractor = Windower(**params)
            elif method_name == 'RelativeLogPower':
                extractor = RelativeLogPower(**params)
            elif method_name == 'Coherences':
                extractor = Coherences(**params)
            elif method_name == 'AsymmetryFeatures':
                extractor = AsymmetryFeatures(**params)
            elif method_name == 'AutoCorrMat':
                extractor = AutoCorrMat(**params)
            elif method_name == 'ARError':
                extractor = ARError(**params)
            elif method_name == 'rqa_channel':
                extractor = rqa_channel(**params)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            steps.append((method_name, extractor))
    
    return Pipeline(steps)

# ==================== 特征提取主函数 ====================

def extract_features_from_window(data, label, pipeline):
    """
    从单个窗口提取特征
    data: (n_channels, n_samples) at 256Hz
    
    注意：如果Windower切分子窗口，会返回多行特征，每行对应一个子窗口
    例如：60秒输入 + window=20 → 返回3行特征，标签都是原始label
    """
    try:
        # 流水线处理
        features = pipeline.transform(np.array([data]))
        
        # 展平特征
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # 检查NaN
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, 0)
        
        # 返回：可能是多行特征（如果有子窗口切分）
        # features.shape[0] = 子窗口数量（如果window=60则为1，window=20则为3）
        n_subwindows = features.shape[0]
        
        # 为每个子窗口复制标签
        labels = [label] * n_subwindows
        valids = [True] * n_subwindows
        
        return features, labels, valids
        
    except Exception as e:
        print(f"⚠️ 特征提取错误: {e}")
        return None, None, None

def process_patient(patient_file, feature_configs, output_dir, srate=256):
    """处理单个患者的数据"""
    
    patient_id = os.path.basename(patient_file).split('_')[0]
    print(f"\n{'='*60}")
    print(f"处理患者: {patient_id}")
    print(f"{'='*60}")
    
    # 加载患者数据
    with open(patient_file, 'rb') as f:
        patient_data = pickle.load(f)
    
    samples = patient_data['samples']
    stats = patient_data['stats']
    
    print(f"总样本数: {len(samples)}")
    print(f"Preictal: {stats['preictal_windows']}")
    print(f"Interictal: {stats['interictal_windows']}")
    
    # 对每种特征配置进行提取
    for feat_name, feat_config in feature_configs.items():
        print(f"\n提取特征: {feat_name}")
        
        # 构建流水线
        pipeline = build_feature_pipeline(feat_config, srate=srate)
        
        # 提取特征
        all_features = []
        all_labels = []
        valid_mask = []
        
        for data, label in tqdm(samples, desc=f"  {feat_name}"):
            feats, lbls, valids = extract_features_from_window(data, label, pipeline)
            
            if feats is not None and len(valids) > 0:
                # 可能返回多行特征（如果有子窗口切分）
                for feat, lbl, valid in zip(feats, lbls, valids):
                    if valid:
                        all_features.append(feat)
                        all_labels.append(lbl)
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
            else:
                valid_mask.append(False)
        
        if len(all_features) == 0:
            print(f"  ⚠️ 没有有效特征，跳过 {feat_name}")
            continue
        
        # 转换为数组
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"  特征形状: {X.shape}")
        print(f"  标签分布: Preictal={np.sum(y==1)}, Interictal={np.sum(y==0)}")
        
        # 创建DataFrame
        feature_columns = [f"{feat_name}_feat{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_columns)
        df['y'] = y
        df['patient'] = patient_id
        df['valid'] = True
        
        # 保存
        feat_output_dir = os.path.join(output_dir, feat_name)
        os.makedirs(feat_output_dir, exist_ok=True)
        
        output_file = os.path.join(feat_output_dir, f"{patient_id}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(df, f, protocol=4)
        
        print(f"  ✅ 保存到: {output_file}")

# ==================== 主程序 ====================

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input_dir",
                      help="输入目录（包含患者pkl文件）")
    parser.add_option("-o", "--output", dest="output_dir",
                      help="输出目录")
    parser.add_option("-c", "--config", dest="config",
                      default="feature_config.yml",
                      help="特征配置文件")
    parser.add_option("-p", "--patient", dest="patient",
                      default="all",
                      help="患者ID或'all'")
    parser.add_option("-n", "--njobs", dest="njobs",
                      default=1, type=int,
                      help="并行任务数")
    
    (options, args) = parser.parse_args()
    
    print("="*60)
    print("CHB-MIT特征提取")
    print("="*60)
    
    # 加载配置
    with open(options.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    srate = config.get('srate', 256)
    
    # 构建特征配置字典
    feature_configs = {}
    for feat_name, feat_config in zip(config['feature_output'], config['features']):
        feature_configs[feat_name] = feat_config
    
    print(f"\n特征配置: {list(feature_configs.keys())}")
    print(f"采样率: {srate} Hz")
    
    # 获取患者文件
    if options.patient == 'all':
        patient_files = sorted([
            os.path.join(options.input_dir, f) 
            for f in os.listdir(options.input_dir) 
            if f.endswith('_continuous.pkl')
        ])
    else:
        # 支持逗号分隔的患者列表
        patient_ids = [p.strip() for p in options.patient.split(',')]
        patient_files = [
            os.path.join(options.input_dir, f"{pid}_continuous.pkl")
            for pid in patient_ids
        ]
    
    print(f"\n找到 {len(patient_files)} 个患者文件")
    
    # 创建输出目录
    os.makedirs(options.output_dir, exist_ok=True)
    
    # 处理患者
    if options.njobs > 1:
        Parallel(n_jobs=options.njobs)(
            delayed(process_patient)(
                pfile, feature_configs, options.output_dir, srate
            ) for pfile in patient_files
        )
    else:
        for pfile in patient_files:
            process_patient(pfile, feature_configs, options.output_dir, srate)
    
    print("\n" + "="*60)
    print("✅ 特征提取完成！")
    print("="*60)

if __name__ == "__main__":
    main()