import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from model_improved import *
import os
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.ndimage as ndi



def peak_shift_rmse(reference,
                    target,
                    raman_shift,
                    prominence=0.02,
                    tol=12,
                    distance_bins=3,
                    min_width=1,
                    return_peaks=False):
    """
    计算峰位RMSE

    """
    ref = np.asarray(reference).ravel()
    tgt = np.asarray(target).ravel()
    x   = np.asarray(raman_shift).ravel()


    idx_ref, _ = find_peaks(ref, prominence=prominence, distance=distance_bins, width=min_width)
    idx_tgt, _ = find_peaks(tgt, prominence=prominence, distance=distance_bins, width=min_width)
    if len(idx_ref) == 0 or len(idx_tgt) == 0:
        out = (np.nan, [])
        if return_peaks:
            return out + ((idx_ref, x[idx_ref]), (idx_tgt, x[idx_tgt]))
        return out

    xr = x[idx_ref][:, None]   # [Nr,1]
    xt = x[idx_tgt][None, :]   # [1,Nt]


    D = np.abs(xr - xt)
    BIG = 1e6
    D_masked = D.copy()
    D_masked[D > tol] = BIG
    r_ind, c_ind = linear_sum_assignment(D_masked)
    keep = D_masked[r_ind, c_ind] < BIG
    r_ind, c_ind = r_ind[keep], c_ind[keep]

    if len(r_ind) == 0:
        out = (np.nan, [])
        if return_peaks:
            return out + ((idx_ref, x[idx_ref]), (idx_tgt, x[idx_tgt]))
        return out

    ref_pos = x[idx_ref[r_ind]]
    tgt_pos = x[idx_tgt[c_ind]]
    rmse = float(np.sqrt(np.mean((ref_pos - tgt_pos) ** 2)))
    matched = list(zip(ref_pos.tolist(), tgt_pos.tolist()))

    if return_peaks:
        return rmse, matched, (idx_ref, x[idx_ref]), (idx_tgt, x[idx_tgt])
    return rmse, matched





def rolling_min_filter(signal, window_size=11):
    padded = np.pad(signal, (window_size // 2,), mode='edge')
    return np.array([
        np.min(padded[i:i + window_size])
        for i in range(len(signal))
    ])


def test_csv(path_source, path_target, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    print("测试函数开始...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 先加载数据以获取真实的 raman_shift 和 reference_spectrum
    data_source = pd.read_csv(path_source, header=0)
    data_target = pd.read_csv(path_target, header=0, encoding='gbk')
    
    # 使用与训练时相同的 raman_shift（确保模型结构一致）
    raman_shift = np.linspace(200, 3200, 1500)  # 与训练时保持一致
    raw_Intensity_target = data_target.iloc[:1500, 1].values.astype(float)
    
    # 目标数据归一化作为参考谱
    min_target = np.min(raw_Intensity_target)
    max_target = np.max(raw_Intensity_target)
    reference_spectrum = (raw_Intensity_target - min_target) / (max_target - min_target + 1e-8)
    
    # 创建模型
    model = ImprovedRamanNet(
        classes=2,
        inputlength=len(raman_shift),
        raman_shift=raman_shift,
        reference_spectrum=reference_spectrum
    ).to(device)
    

    # 直接加载新训练的模型权重
    try:
        state_dict = torch.load('save_model/save_schemeA/latest_model.pth', map_location=device, weights_only=True)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("✓ 加载新训练的模型权重成功")
        if missing_keys:
            print(f"缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"意外的键: {unexpected_keys}")
    except FileNotFoundError:
        print("未找到模型文件，使用随机初始化的模型")
        print("请确保已使用修改后的代码训练模型")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        print("使用随机初始化的模型")
    
    model.eval()

    # 加载源数据
    raw_Intensity_source = data_source.iloc[:1500, 1].values.astype(float)
    # 获取真实的拉曼位移轴用于可视化
    real_raman_shift = data_source.iloc[:1500, 0].values.astype(float)

    # —— 统一归一化基准：目标的 min–max ——
    t_min, t_max = np.min(raw_Intensity_target), np.max(raw_Intensity_target)
    if t_max - t_min < 1e-6: t_max = t_min + 1.0
    def norm(y): return (y - t_min) / (t_max - t_min + 1e-8)

    normalized_Intensity_target = norm(raw_Intensity_target)
    normalized_Intensity_source = norm(raw_Intensity_source)
    print(f"源数据归一化范围: {np.min(normalized_Intensity_source):.4f}-{np.max(normalized_Intensity_source):.4f}")
    print(f"目标数据归一化范围: {np.min(normalized_Intensity_target):.4f}-{np.max(normalized_Intensity_target):.4f}")

    # 模型输入
    aligned_src = normalized_Intensity_source

    Intensity_tensor = torch.FloatTensor(aligned_src).unsqueeze(0).to(device)

    with torch.no_grad():
        class_out, trans_tensor, domain_coeff,_ ,_= model(Intensity_tensor)  # 现在返回三个值
        trans_intensity = trans_tensor.cpu().numpy().flatten()
        _, predicted = torch.max(class_out, 1)
        if predicted.item() == 0:
            material = "对乙酰氨基酚"
        elif predicted.item() == 1:
            material = "环己烷"
        else:
            material = "萘"

        print(f"预测物质: {material}")



    # 迁移后归一化 - 使用统一基准
    normalized_trans_intensity = norm(trans_intensity)

    # 平滑处理
    wl, po = 11, 3
    sm_source = savgol_filter(normalized_Intensity_source, wl, po)
    sm_target = savgol_filter(normalized_Intensity_target, 11, 3)
    sm_trans = savgol_filter(normalized_trans_intensity, 11, 3)
            # 0* rolling_min_filter(normalized_trans_intensity, 11) + \
    with torch.no_grad():
        # 源光谱与目标光谱的相似度（迁移前）
        similarity_before = torch.cosine_similarity(
            torch.FloatTensor(normalized_Intensity_source).unsqueeze(0),
            torch.FloatTensor(normalized_Intensity_target).unsqueeze(0)
        ).item()

        # 迁移后光谱与目标光谱的相似度
        similarity_after = torch.cosine_similarity(
            torch.FloatTensor(sm_trans).unsqueeze(0),
            torch.FloatTensor(sm_target).unsqueeze(0)
        ).item()

    print(f"迁移前相似度: {similarity_before:.4f}, 迁移后相似度: {similarity_after:.4f}")

    # 计算峰位偏移 RMSE
    print("计算峰位偏移RMSE...")
    rmse_before, matched_before = peak_shift_rmse(normalized_Intensity_source,
                                                  normalized_Intensity_target,
                                                  real_raman_shift)
    print(f"迁移前 RMSE: {rmse_before:.2f} cm⁻¹, 匹配峰数量: {len(matched_before)}")
    rmse_after, matched_after = peak_shift_rmse(sm_trans, sm_target, real_raman_shift)

    print(f"迁移后 RMSE: {rmse_after:.2f} cm⁻¹, 匹配峰数量: {len(matched_after)}")

    eps = 1e-8

    eff_coeff_raw = normalized_trans_intensity / (normalized_Intensity_source + eps)
    eff_coeff_smooth = sm_trans / (sm_source + eps)

    eff_vis = np.clip(eff_coeff_raw, 0.0, 3.0)  # 原始
    eff_vis_smooth = np.clip(eff_coeff_smooth, 0.0, 3.0)  # 平滑




    # 可视化
    print("创建可视化图表...")
    plt.figure(figsize=(15, 9))

    # # 1. 迁移前归一化对比
    plt.subplot(2, 2, 1)
    plt.plot(real_raman_shift, normalized_Intensity_source, 'b-', label='Source (Normalized)')
    plt.plot(real_raman_shift, normalized_Intensity_target, 'g-', label='Target (Normalized)', alpha=0.7)
    plt.title(f'Before Transfer\nSimilarity: {similarity_before:.3f}, RMSE: {rmse_before:.2f} cm⁻¹')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Normalized Intensity')
    plt.legend()

    # 2. 域系数 vs 有效因子
    plt.subplot(2, 2, 2)
    domain_coeff_np = domain_coeff.detach().cpu().numpy().flatten()
    
    plt.plot(real_raman_shift, domain_coeff_np, 'm-',  label='Domain Coefficients')
    plt.plot(real_raman_shift, eff_vis,          '-',  label='Effective coeff (trans/source)', alpha=0.7)
    # plt.plot(real_raman_shift, eff_vis_smooth,  '--', label='Effective coeff (smoothed)', alpha=0.7)

    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.4)
    plt.title('Domain Coefficients vs Effective Multiplier')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Value')
    plt.legend()

    # # 3. 迁移后对比
    plt.subplot(2, 2, 3)
    plt.plot(real_raman_shift, normalized_trans_intensity, 'r-', label='Transferred (Raw)')
    plt.plot(real_raman_shift, normalized_Intensity_target, 'g-', label='Target (Normalized)', alpha=0.7)
    plt.title('After Transfer (Raw)')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Normalized Intensity')
    plt.legend()


    # 4. 迁移后 vs 目标（平滑后）
    plt.subplot(2, 2, 4)
    plt.plot(real_raman_shift, sm_trans, 'r-', label='Transferred (Smoothed)')
    plt.plot(real_raman_shift, sm_target, 'g-', label='Target (Normalized)', alpha=0.7)
    plt.title(f'After Transfer (Smoothed)\nSimilarity: {similarity_after:.3f}, RMSE: {rmse_after:.2f} cm⁻¹')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Normalized Intensity')
    plt.legend()

    save_path = os.path.join(save_dir, 'comparison_test.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"测试完成，结果已保存至: {save_path}")

    return {
        'raman_shift': real_raman_shift,
        'source_raw': raw_Intensity_source,
        'source_normalized': normalized_Intensity_source,
        'target_raw': raw_Intensity_target,
        'target_normalized': normalized_Intensity_target,
        'transferred_raw': trans_intensity,
        'transferred_normalized': normalized_trans_intensity,
        'domain_coeff': domain_coeff.detach().cpu().numpy().flatten(),
        'similarity_before': similarity_before,
        'similarity_after': similarity_after,
        'rmse_before': rmse_before,
        'rmse_after': rmse_after,
        'material': material
    }


if __name__ == "__main__":
    results = test_csv(
    path_target = r"",
    path_source= r""
    )

    print(f"预测物质: {results['material']}")
    print(f"迁移前相似度: {results['similarity_before']:.4f}")
    print(f"迁移后相似度: {results['similarity_after']:.4f}")
    print(f"迁移前 RMSE: {results['rmse_before']:.2f} cm⁻¹")
    print(f"迁移后 RMSE: {results['rmse_after']:.2f} cm⁻¹")
