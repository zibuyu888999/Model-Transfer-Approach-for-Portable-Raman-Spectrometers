
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from model_improved import *
import os
from scipy.signal import find_peaks, savgol_filter
import logging
import random
import numpy as np
import scipy.ndimage as ndi
from visualization import visualize_alignment, visualize_alignment_3
from losses import (
    _soft_peak_map_physical, peak_alignment_loss_soft, 
    segment_peak_alignment_loss, segment_similarity_loss_zcorr,
    interface_continuity_loss, global_similarity_loss_emd_simse,
     simse_loss, emd_simse_loss, spectral_correlation_loss,
    masked_cosine_similarity,  intensity_matching_loss,
    segment_intensity_l2_loss, segment_intensity_ratio_loss,extract_segments_aligned,compute_phase_a_losses
)

# 设置字体，优先使用支持特殊字符的字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def peak_response(y, sigma=3):
    """1D 峰性响应：高斯平滑 + 拉普拉斯（LoG近似），返回非负曲线"""
    y_s = ndi.gaussian_filter1d(y, sigma=sigma, mode='nearest')
    lap = ndi.gaussian_laplace(y_s, sigma=sigma, mode='nearest')
    resp = np.maximum(-lap, 0.0)
    return resp


def build_peak_prior_from_loader(train_loader, max_batches=50, sigma=3, trim_q=0.1, device="cpu"):
    """从训练集中若干条原始光谱构建通用峰性先验 P_all(k) ∈ [0,1]"""
    Rs = []
    n_batches = 0
    for inputs, _ in train_loader:
        x = inputs.detach().to("cpu").numpy()  # 假设 inputs: [B, L]
        for y in x:
            Rs.append(peak_response(y, sigma=sigma))
        n_batches += 1
        if n_batches >= max_batches:
            break
    R = np.stack(Rs, axis=0)  # [N, L]
    # 截尾均值（鲁棒），避免个别样本异常影响
    low, high = np.quantile(R, [trim_q, 1 - trim_q], axis=0)
    R_clip = np.clip(R, low, high)
    P_all = R_clip.mean(axis=0)  # [L]
    # 归一到 [0,1]
    P_all = (P_all - P_all.min()) / (P_all.max() - P_all.min() + 1e-8)
    return P_all.astype(np.float32)

# ===== 轻量级光谱在线增强（仅训练集用） =====
def augment_spectrum(y: np.ndarray) -> np.ndarray:
    L = y.shape[0]
    x = np.linspace(-1, 1, L)

    # 1) 强度缩放（±10%）
    a = np.random.uniform(0.9, 1.1)
    y1 = a * y

    # 2) 低阶基线漂移（≤3%峰高）
    c0 = np.random.uniform(-0.01, 0.01)
    c1 = np.random.uniform(-0.01, 0.01)
    c2 = np.random.uniform(-0.005, 0.005)
    baseline = c0 + c1 * x + c2 * (x ** 2)
    y2 = y1 + baseline

    # 3) 微小波数整体平移（±2 bins）
    shift_bins = np.random.randint(-2, 3)
    if shift_bins != 0:
        idx = np.clip(np.arange(L) + shift_bins, 0, L - 1)
        y3 = y2[idx]
    else:
        y3 = y2

    # 4) 峰宽轻微变化（与窄高斯核做卷积，≤10%）
    try:
        from scipy.ndimage import gaussian_filter1d
        sigma = np.random.uniform(0.0, 1.0)   # 0~1点的平滑
        y4 = gaussian_filter1d(y3, sigma=sigma, mode='nearest')
    except Exception:
        y4 = y3

    # 5) 小噪声（≤2%峰高）
    noise = np.random.normal(0.0, 0.02 * (y4.max() - y4.min() + 1e-8), size=L)
    y5 = y4 + noise

    # 保持到 [0,1]（和你数据预处理一致）
    y5 = (y5 - y5.min()) / (y5.max() - y5.min() + 1e-8)
    return y5.astype(np.float32)


class AugmentDataset(Dataset):
    """把 TensorDataset 包装成带在线增强的数据集；仅 train=True 时生效。"""
    def __init__(self, X: torch.Tensor, y: torch.Tensor, train: bool = True):
        self.X = X  # [N, L]
        self.y = y  # [N]
        self.train = train

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].cpu().numpy()
        if self.train:
            x = augment_spectrum(x)
        else:
            x = x.astype(np.float32)
        return torch.from_numpy(x), self.y[idx]



def load_data(path):
    portable_ace = np.load('/home/yutingli/claude/claude/大拉曼_对乙酰氨基酚.npy')
    portable_cyc = np.load('/home/yutingli/claude/claude/大拉曼_环己烷.npy')
    portable_nap = np.load('/home/yutingli/claude/claude/大拉曼_萘.npy') 

    handheld_ace = np.load('/home/yutingli/claude/claude/手持式_对乙酰氨基酚')
    handheld_cyc = np.load('/home/yutingli/claude/claude/手持式_环己烷.npy')
    handheld_nap = np.load('/home/yutingli/claude/claude/手持式_萘.npy') 

    # 分别处理源数据和目标数据，避免混合归一化
    # 目标数据归一化
    target_data = np.vstack([portable_ace, portable_cyc, portable_nap])
    target_min, target_max = target_data.min(), target_data.max()
    target_data = (target_data - target_min) / (target_max - target_min + 1e-8)
    
    # 源数据归一化
    source_data = np.vstack([handheld_ace, handheld_cyc, handheld_nap])
    source_min, source_max = source_data.min(), source_data.max()
    source_data = (source_data - source_min) / (source_max - source_min + 1e-8)
    
    # 重新切分数据
    n_pa, n_pc, n_pn = len(portable_ace), len(portable_cyc), len(portable_nap)
    n_ha, n_hc, n_hn = len(handheld_ace), len(handheld_cyc), len(handheld_nap)
    
    portable_ace = target_data[:n_pa]
    portable_cyc = target_data[n_pa:n_pa + n_pc]
    portable_nap = target_data[n_pa + n_pc:n_pa + n_pc + n_pn]
    
    handheld_ace = source_data[:n_ha]
    handheld_cyc = source_data[n_ha:n_ha + n_hc]
    handheld_nap = source_data[n_ha + n_hc:]

    # === 标签改成三类 (0=对乙酰氨基酚, 1=环己烷, 2=萘) ===
    target_data = np.vstack([portable_ace, portable_cyc, portable_nap])
    target_labels = np.concatenate([
        np.zeros(len(portable_ace)),
        np.ones(len(portable_cyc)),
        np.full(len(portable_nap), 2)
    ])

    source_data = np.vstack([handheld_ace, handheld_cyc, handheld_nap])
    source_labels = np.concatenate([
        np.zeros(len(handheld_ace)),
        np.ones(len(handheld_cyc)),
        np.full(len(handheld_nap), 2)
    ])

    # 训练/测试集划分保持不变
    X_train, X_test, y_train, y_test = train_test_split(
        source_data, source_labels, test_size=0.2, stratify=source_labels, random_state=42)

    X_domain_train, X_domain_test, y_domain_train, y_domain_test = train_test_split(
        target_data, target_labels, test_size=0.2, stratify=target_labels, random_state=42)

    batch_size = 64

    # === 训练/测试集 DataLoader（只改训练集） ===
    from torch.utils.data import WeightedRandomSampler

    # 在线增强数据集
    train_dataset = AugmentDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train), train=True)
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    # 类权重（按训练集频次的反比）
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(y_train),
        replacement=True
    )

    # 注意：训练集用 sampler，禁用 shuffle；其它保持不变
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    domain_train_dataset = TensorDataset(torch.FloatTensor(X_domain_train), torch.LongTensor(y_domain_train))
    domain_test_dataset = TensorDataset(torch.FloatTensor(X_domain_test), torch.LongTensor(y_domain_test))
    domain_train_loader = DataLoader(domain_train_dataset, batch_size, shuffle=True, drop_last=True)
    domain_test_loader = DataLoader(domain_test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader, domain_train_loader, domain_test_loader, torch.FloatTensor(class_weights)




def train():
    # --- 基础设置 ---
    os.makedirs('save_model', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True


    train_loader, test_loader, domain_train_loader, _, class_weights = load_data('./')
    raman_shift = np.linspace(200, 3200, 1500) # 

    # --- 用训练集构建通用峰性先验 P_all（你已有的做法） ---
    P_all = build_peak_prior_from_loader(train_loader, max_batches=50, sigma=3, trim_q=0.1, device=device)
    model = ImprovedRamanNet(raman_shift=raman_shift, reference_spectrum=P_all).to(device)

    # ---------------- 可选：注册 AttentionGate 钩子（仅诊断，可视化中间特征） ----------------

    # 先冻结强度变换器，5 epoch 后再解冻
    for p in model.intensity_transformer.parameters():
        p.requires_grad = True

    # 更新XAxisWarp的最大偏移范围
    model.xwarp.max_shift = 25.0  # 进一步增加最大偏移范围



    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # criterion_class = nn.CrossEntropyLoss()
    criterion_class = nn.CrossEntropyLoss(weight=class_weights.to(device))

    init_acc = []
    init_loss_class, init_loss_domain1 = [], []
    init_loss_rmse, init_loss_gs = [], []
    best_acc = 0.0


    PEAK_FREQ = 1
    global_step = 0

    # ---- 三阶段损失权重 ----
    def get_weights(epoch):
        def lin(a, b, t0, t1):
            if epoch <= t0: return a
            if epoch >= t1: return b
            α = (epoch - t0) / (t1 - t0)
            return a + α*(b - a)


        if epoch < 50:  # Phase A: 分段快速对齐
                return dict(
                cls=0.03,                
                d1=0.1,                  
                peak=0.00, peak_align=0.08,
                segment_peak=0.3,        # 段内软峰对齐
                segment_sim=0.2,         # 段内相似（z-corr/导数）
                interface_continuity=0.2, # 段间连续性损失
                segment_intensity=0.2,       # 段内强度
                segment_intensity_ratio=0.25,
                corr=0.08, gs=0.06, l1=0.05  # 留形状锚点
            )   # ≈ 1.50

        elif epoch < 130:  # Phase B: 整谱对齐主导
                return dict(
                cls=0.10,
                d1=0.20,                 #
                peak=0.00,               # 
                peak_align=0.01,         # 
                global_sim=0.65,          # 主力：EMD+SIMSE
                segment_peak=0.06,
                segment_sim=0.08,
                interface_continuity=0.2,
                segment_intensity=0.08,
                segment_intensity_ratio=0.08,
                corr=0.08, gs=0.08, l1=0.05
            )   # ≈ 1.48–1.53

        else:  # Phase C: 精修
                return dict(
                cls=lin(0.15, 0.20, 65, 75),   
                d1=0.20,
                peak=0.00,                     
                peak_align=0.05,
                interface_continuity=0.4,             # 段间连续性损失
                global_sim=0.3,               # 全局相似度损失
                segment_peak=0.06,
                segment_sim=0.06,
                segment_intensity=0.08, segment_intensity_ratio=0.08,
                corr=0.22, gs=0.12, l1=0.12
            )   # ≈ 1.50


    def peak_sched(epoch: int):

        if epoch < 70:
            return dict(
                tau=0.13,             
                thresh_k=1.4,         
                radius_bins=35,       
                lam=1.6,              
                sink_mass=1e-4,       
                sinkhorn_eps=0.5      
            )

        elif epoch < 110:
            return dict(
                tau=0.10, thresh_k=1.8,
                radius_bins=32, lam=1.3,
                sink_mass=5e-4, sinkhorn_eps=0.25
            )

        else:
            return dict(
                tau=0.08, thresh_k=2.2,
                radius_bins=30, lam=1.25,
                sink_mass=1e-3, sinkhorn_eps=0.20
            )

    for epoch in range(150):
        if epoch == 5:
            for p in model.intensity_transformer.parameters():
                p.requires_grad = True

        model.train()
        domain_iter = iter(domain_train_loader)
        w = get_weights(epoch)

        total_loss_c = total_loss_d1 = total_loss_r = total_loss_gs = 0.0
        total_loss_peak_align = total_loss_corr = 0.0
        # Phase A 损失统计
        total_loss_segment_peak = total_loss_segment_sim = total_loss_interface_continuity = 0.0
        total_loss_segment_intensity, total_loss_segment_intensity_ratio = 0.0, 0.0
        total_loss_global_sim = 0.0
        total_samples = 0

 
        torch.autograd.set_detect_anomaly(True) 

        def _chk(name, t):
            if isinstance(t, torch.Tensor):
                if not torch.isfinite(t).all():
                    bad = t[~torch.isfinite(t)]
                    print(f"[BAD] {name}: has {bad.numel()} non-finite, "
                          f"min={float(torch.nanmin(t))}, max={float(torch.nanmax(t))}")
                    return False
            return True

        for inputs, labels in train_loader:
            
            try:
                domain_inputs, domain_labels = next(domain_iter)
            except StopIteration:
                domain_iter = iter(domain_train_loader)
                domain_inputs, domain_labels = next(domain_iter)

            inputs = inputs.to(device)  # [B,L]
            labels = labels.to(device)
            domain_inputs = domain_inputs.to(device)
            domain_labels = domain_labels.to(device)

            # —— 样本自适应软掩码 M(k) —— #
            with torch.no_grad():
                # 峰性响应 R_y(k)
                x_np = inputs.detach().to("cpu").numpy()
                resp_list = []
                for y in x_np:
                    r = peak_response(y, sigma=3)
                    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
                    resp_list.append(r)
                Rb = torch.tensor(np.stack(resp_list, 0), device=device, dtype=torch.float32)  # [B,L]


                P = torch.tensor(P_all, device=device, dtype=torch.float32).unsqueeze(0).expand_as(Rb)  # [B,L]
                gamma = 0.7
                M = (P ** gamma) * (Rb ** (1.0 - gamma))
                M = 0.2 + 0.8 * M

                M_bin = (M > torch.quantile(M, 0.8, dim=1, keepdim=True)).float()
                mask_union = (M_bin.max(dim=0).values > 0.5).float()  # [L]
                mask_each = M_bin 

            with torch.no_grad():

                B, L = inputs.shape
                device, dtype = inputs.device, inputs.dtype
                domain_matched_list = []

                for y in labels.tolist():
                        cand = domain_inputs[domain_labels == y]
                        if cand.shape[0] == 0:
                            cand = domain_inputs  # 回退

                        s_n    = F.normalize(inputs[labels == y][0], dim=0)  # 当前这条样本
                        cand_n = F.normalize(cand, dim=1)

                        # 余弦相似度（向量化）
                        sim = cand_n @ s_n              # [Nc]
                        j = torch.argmax(sim)
                        domain_matched_list.append(cand[j])

                domain_matched = torch.stack(domain_matched_list, dim=0).to(device=device, dtype=dtype)
                assert _chk("domain_matched", domain_matched)
            
            # ---------- 单步前向 ----------
            class_out, trans_A2B, domain_coeff, x_gated_smoothed, segment_chunks = model(inputs)  # 现在返回五个值: [B,L], [B,L], [B,L], [B,L], [B,K,M]
            

            assert _chk("trans_A2B", trans_A2B)

            trans_init_A2B = inputs  # 与参数无关

            # ---------- 各项损失 ----------
            # 分类
            loss_class = criterion_class(class_out, labels)

            # 迁移-相似度（对 trans_A2B 反传梯度）
            loss_domain1 = masked_cosine_similarity(trans_A2B, domain_matched, M)


            # 峰位 EMD 损失：使用新的 peak_emd_loss 函数
            use_peak = (global_step % PEAK_FREQ == 0)
            if use_peak:
                raman_shift_tensor = torch.tensor(raman_shift, device=device, dtype=torch.float32)

                loss_peak = torch.tensor(0.0, device=device)
            else:
                loss_peak = torch.tensor(0.0, device=device)

            # 梯度结构 & L1（逐点加权，可反传）
            grad_pred = trans_A2B[:, 1:] - trans_A2B[:, :-1]
            grad_tgt = domain_matched[:, 1:] - domain_matched[:, :-1]
            gs_mask = mask_each[...,1:] * mask_each[...,:-1]
            gs_loss = (gs_mask * (grad_pred - grad_tgt).abs()).sum() / (gs_mask.sum() + 1e-8)


            l1_mask = mask_each                                                     # [B,L]
            l1_loss = (l1_mask * (trans_A2B - domain_matched).abs()).sum() / (l1_mask.sum() + 1e-8)

            # 新增损失函数 - 使用可导的软峰概率图版本
            # loss_peak_align = peak_alignment_loss_soft(trans_A2B, domain_matched, raman_shift)
            cfg = peak_sched(epoch)
            loss_peak_align = peak_alignment_loss_soft(
                    trans_A2B, domain_matched, raman_shift,
                    **cfg,                    # 分段参数
                    w_emd=1.0, w_simse=0.5  
                )

            loss_corr = spectral_correlation_loss(trans_A2B, domain_matched)


            loss_global_sim = torch.tensor(0.0, device=device)
            tau_used = None
            if 55 <= epoch:
                loss_global_sim, parts_global_sim, tau_used = global_similarity_loss_emd_simse(
                    trans_A2B,           # [B,L]
                    domain_matched,      # [B,L]
                    raman_shift,         # [L] 或 tensor
                    epoch=epoch,
                    phase_start=35, phase_end=150,
                    tau_start=0.12, tau_end=0.08,   # 先软后尖
                    w_emd=1.5, w_simse=0.5          # 主力靠EMD，SIMSE 降权
                )

            # ---------- Phase A: 分段快速对齐损失 ----------
            phase_a_losses = compute_phase_a_losses(
                trans_A2B,                        # [B,L]
                domain_matched,                   # [B,L]
                raman_shift,                     
                M=300, O=150,
                segment_chunks=segment_chunks    
            )

            # ---------- 按权重汇总总损失（单步反传） ----------
            def W(name):  
                return w.get(name, 0.0)

            total_loss = (
                    W('cls')        * loss_class +
                    W('d1')         * loss_domain1 +
                    W('peak')       * loss_peak +
                    W('peak_align') * loss_peak_align +
                    W('corr')       * loss_corr +
                    W('gs')         * gs_loss +
                    W('l1')         * l1_loss +

                    # Phase A（统一从 phase_a_losses 取）
                    W('segment_peak')            * phase_a_losses['segment_peak'] +
                    W('segment_sim')             * phase_a_losses['segment_sim'] +
                    W('interface_continuity')    * phase_a_losses['interface_continuity'] +
                    W('segment_intensity_ratio') * phase_a_losses['segment_intensity_ratio'] +
                    W('segment_intensity')       * phase_a_losses['segment_intensity'] +

                    # 全局相似（分阶段启用）
                    W('global_sim') * loss_global_sim
            )
            # 形变正则
            total_loss = total_loss + 0.05 * model.xwarp.smooth_l1 + 0.08 * model.xwarp.smooth_l2



            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # 统计
            bsz = inputs.size(0)
            total_samples += bsz
            total_loss_c += loss_class.item() * bsz
            total_loss_d1 += loss_domain1.item() * bsz
            total_loss_r += loss_peak.item() * bsz
            total_loss_gs += gs_loss.item() * bsz
            total_loss_peak_align += loss_peak_align.item() * bsz
            total_loss_corr += loss_corr.item() * bsz
            
            # Phase A 损失统计
            total_loss_segment_peak += phase_a_losses['segment_peak'].item() * bsz
            total_loss_segment_sim += phase_a_losses['segment_sim'].item() * bsz
            total_loss_interface_continuity += phase_a_losses['interface_continuity'].item() * bsz
            total_loss_segment_intensity += phase_a_losses['segment_intensity'].item() * bsz
            total_loss_segment_intensity_ratio += phase_a_losses['segment_intensity_ratio'].item() * bsz
            
            # 全局相似度损失统计
            total_loss_global_sim += loss_global_sim.item() * bsz

            global_step += 1

        # ------- epoch 级日志与评估 -------
        epoch_loss_c = total_loss_c / max(total_samples, 1)
        epoch_loss_d1 = total_loss_d1 / max(total_samples, 1)
        epoch_loss_r = total_loss_r / max(total_samples, 1)
        epoch_loss_gs = total_loss_gs / max(total_samples, 1)
        epoch_loss_peak_align = total_loss_peak_align / max(total_samples, 1)
        epoch_loss_corr = total_loss_corr / max(total_samples, 1)
        
        # Phase A 损失统计
        epoch_loss_segment_peak = total_loss_segment_peak / max(total_samples, 1)
        epoch_loss_segment_sim = total_loss_segment_sim / max(total_samples, 1)
        epoch_loss_interface_continuity = total_loss_interface_continuity / max(total_samples, 1)
        epoch_loss_global_sim = total_loss_global_sim / max(total_samples, 1)
        epoch_loss_segment_intensity = total_loss_segment_intensity / max(total_samples, 1)
        epoch_loss_segment_intensity_ratio = total_loss_segment_intensity_ratio / max(total_samples, 1)

        test_acc = test(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'save_model/best_model.pth')
        torch.save(model.state_dict(), 'save_model/latest_model.pth')

        init_acc.append(test_acc)
        init_loss_class.append(epoch_loss_c)
        init_loss_domain1.append(epoch_loss_d1)
        init_loss_rmse.append(epoch_loss_r)
        init_loss_gs.append(epoch_loss_gs)

        # 根据阶段打印不同的损失信息
        if epoch < 65:  # Phase A
            print(f"Epoch {epoch + 1:03d} | Acc: {test_acc:.4f} | "
                  f"Class: {epoch_loss_c:.4f} | D1: {epoch_loss_d1:.4f} | "
                  f"SegPeak: {epoch_loss_segment_peak:.4f} | SegSim: {epoch_loss_segment_sim:.4f} | Interface: {epoch_loss_interface_continuity:.4f} | "
                  f"SegInt: {epoch_loss_segment_intensity:.4f} | SegRatio: {epoch_loss_segment_intensity_ratio:.4f}")

        #           f"SegPeak: {epoch_loss_segment_peak:.4f} | SegSim: {epoch_loss_segment_sim:.4f} | Interface: {epoch_loss_interface_continuity:.4f}")
        elif epoch < 110:  # Phase C (45-80) - 全局相似度启用
            print(f"Epoch {epoch + 1:03d} | Acc: {test_acc:.4f} | "
              f"Class: {epoch_loss_c:.4f} | D1: {epoch_loss_d1:.4f} | "
            #   f"RMSE: {epoch_loss_r:.4f} | "
              f"PeakAlign: {epoch_loss_peak_align:.4f} | Corr: {epoch_loss_corr:.4f} | "
                  f"GS: {epoch_loss_gs:.4f} | GlobalSim: {epoch_loss_global_sim:.4f}")
        else:  # Phase C (45+)
            print(f"Epoch {epoch + 1:03d} | Acc: {test_acc:.4f} | "
              f"Class: {epoch_loss_c:.4f} | D1: {epoch_loss_d1:.4f} | "
            #   f"RMSE: {epoch_loss_r:.4f} | "
              f"PeakAlign: {epoch_loss_peak_align:.4f} | Corr: {epoch_loss_corr:.4f} | "
                  f"GS: {epoch_loss_gs:.4f} | GlobalSim: {epoch_loss_global_sim:.4f}")

    return init_acc, init_loss_class, init_loss_domain1, init_loss_rmse, init_loss_gs


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            class_out, _, domain_coeff, _, _ = model(inputs)  # 现在返回五个值
            _, predicted = torch.max(class_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def peak_shift_rmse(reference, target, raman_shift, prominence=0.02):
    reference = np.asarray(reference).flatten()
    target = np.asarray(target).flatten()
    raman_shift = np.asarray(raman_shift).flatten()

    peaks_ref, _ = find_peaks(reference, prominence=prominence)
    peaks_tgt, _ = find_peaks(target, prominence=prominence)

    if len(peaks_ref) == 0 or len(peaks_tgt) == 0:
        return np.nan, []

    peaks_ref_shift = raman_shift[peaks_ref]
    peaks_tgt_shift = raman_shift[peaks_tgt]

    matched = []
    for peak in peaks_ref_shift:
        nearest = min(peaks_tgt_shift, key=lambda x: abs(x - peak))
        matched.append((peak, nearest))

    ref_arr = np.array([x[0] for x in matched])
    tgt_arr = np.array([x[1] for x in matched])
    rmse = np.sqrt(np.mean((ref_arr - tgt_arr) ** 2))
    return rmse, matched


def test_csv(path_source, path_target, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    print("开始测试单条数据...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = ImprovedRamanNet().to(device)
    try:
        model.load_state_dict(torch.load('save_model/latest_model.pth', map_location=device))
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise FileNotFoundError("找不到模型文件，请先训练模型")
    model.eval()



    data = pd.read_csv(path_source, header=None)
    Raman_shift = data.iloc[:1500, 0].values.astype(float)
    raw_Intensity_source = data.iloc[:1500, 1].values.astype(float)

    min_source = np.min(raw_Intensity_source)
    max_source = np.max(raw_Intensity_source)
    normalized_Intensity_source = (raw_Intensity_source - min_source) / (max_source - min_source + 1e-8)
    print(f"源数据归一化范围: {np.min(normalized_Intensity_source):.4f}-{np.max(normalized_Intensity_source):.4f}")



    data_target = pd.read_csv(path_target, header=0)
    raw_Intensity_target = data_target.iloc[:1500, 1].values.astype(float)
    min_target = np.min(raw_Intensity_target)
    max_target = np.max(raw_Intensity_target)
    normalized_Intensity_target = (raw_Intensity_target - min_target) / (max_target - min_target + 1e-8)
    print(f"目标数据归一化范围: {np.min(normalized_Intensity_target):.4f}-{np.max(normalized_Intensity_target):.4f}")

    Intensity_tensor = torch.FloatTensor(normalized_Intensity_source).unsqueeze(0).to(device)

    with torch.no_grad():
        class_out, trans_tensor, domain_coeff, _, _ = model(Intensity_tensor) # 现在返回五个值
        trans_intensity = trans_tensor.cpu().numpy().flatten()
        _, predicted = torch.max(class_out, 1)
        if predicted.item() == 0:
            material = "对乙酰氨基酚"
        elif predicted.item() == 1:
            material = "环己烷"
        else:
            material = "萘"

        print(f"预测物质: {material}")

    min_trans = np.min(trans_intensity)
    max_trans = np.max(trans_intensity)
    normalized_trans_intensity = (trans_intensity - min_trans) / (max_trans - min_trans + 1e-8)
    print(f"迁移数据归一化范围: {np.min(normalized_trans_intensity):.4f}-{np.max(normalized_trans_intensity):.4f}")

    with torch.no_grad():
        similarity_before = torch.cosine_similarity(
            torch.FloatTensor(normalized_Intensity_source).unsqueeze(0),
            torch.FloatTensor(normalized_Intensity_target).unsqueeze(0)
        ).item()

        similarity_after = torch.cosine_similarity(
            torch.FloatTensor(normalized_trans_intensity).unsqueeze(0),
            torch.FloatTensor(normalized_Intensity_target).unsqueeze(0)
        ).item()

    print(f"迁移前相似度: {similarity_before:.4f}, 迁移后相似度: {similarity_after:.4f}")

    rmse_before, _ = peak_shift_rmse(normalized_Intensity_source, normalized_Intensity_target, Raman_shift)
    rmse_after, _ = peak_shift_rmse(normalized_trans_intensity, normalized_Intensity_target, Raman_shift)
    print(f"峰位RMSE(迁移前): {rmse_before:.2f} cm⁻¹, 峰位RMSE(迁移后): {rmse_after:.2f} cm⁻¹")

    # 计算有效因子
    eps = 1e-8
    eff_coeff_raw = normalized_trans_intensity / (normalized_Intensity_source + eps)
    eff_vis = np.clip(eff_coeff_raw, 0.0, 3.0)  # 限制显示范围

    # 平滑处理
    wl, po = 11, 3
    sm_source = savgol_filter(normalized_Intensity_source, wl, po)
    sm_target = savgol_filter(normalized_Intensity_target, wl, po)
    sm_trans = savgol_filter(normalized_trans_intensity, wl, po)

    # 可视化 - 四幅图布局
    print("创建可视化图表...")
    plt.figure(figsize=(15, 9))

    # 1. 迁移前归一化对比
    plt.subplot(2, 2, 1)
    plt.plot(Raman_shift, normalized_Intensity_source, 'b-', label='Source (Normalized)')
    plt.plot(Raman_shift, normalized_Intensity_target, 'g-', label='Target (Normalized)', alpha=0.7)
    plt.title(f'Before Transfer\nSimilarity: {similarity_before:.3f}, RMSE: {rmse_before:.2f} cm⁻¹')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Normalized Intensity')
    plt.legend()

    # 2. 域系数 vs 有效因子
    plt.subplot(2, 2, 2)
    domain_coeff_np = domain_coeff.detach().cpu().numpy().flatten()
    
    # plt.plot(Raman_shift, domain_coeff_np, 'm-',  label='Domain Coefficients')
    plt.plot(Raman_shift, eff_vis,          '-',  label='Effective coeff (trans/source)', alpha=0.7)

    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.4)
    plt.title('Domain Coefficients vs Effective Multiplier')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Value')
    plt.legend()

    # 3. 迁移后对比（未平滑）
    plt.subplot(2, 2, 3)
    plt.plot(Raman_shift, normalized_trans_intensity, 'r-', label='Transferred (Raw)')
    plt.plot(Raman_shift, normalized_Intensity_target, 'g-', label='Target (Normalized)', alpha=0.7)
    plt.title('After Transfer (Raw)')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Normalized Intensity')
    plt.legend()

    # 4. 迁移后 vs 目标（平滑后）
    plt.subplot(2, 2, 4)
    plt.plot(Raman_shift, sm_trans, 'r-', label='Transferred (Smoothed)')
    plt.plot(Raman_shift, sm_target, 'g-', label='Target (Normalized)', alpha=0.7)
    plt.title(f'After Transfer (Smoothed)\nSimilarity: {similarity_after:.3f}, RMSE: {rmse_after:.2f} cm⁻¹')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Normalized Intensity')
    plt.legend()

    save_path = os.path.join(save_dir, 'comparison.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"测试完成，结果已保存至: {save_path}")



    return normalized_trans_intensity, material, similarity_before, similarity_after, rmse_before, rmse_after


if __name__ == "__main__":
    init_acc, init_loss_class, init_loss_domain1, init_loss_rmse, init_loss_gs = train()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = range(1, len(init_loss_class) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs, init_loss_class, label='Classification Loss', color='blue')
    ax1.plot(epochs, init_loss_domain1, label='Domain Loss 1', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(epochs, init_loss_rmse, label='Peak Shift RMSE (log)', color='red')
    ax2.set_ylabel('Peak Shift RMSE (log)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('Model Losses over Epochs with Log-scaled Peak Shift RMSE')
    plt.grid(True)
    plt.savefig('loss_curve_with_log_rmse.png', dpi=300)
    plt.close()

    path_target = r""
    path_source= r""
    trans_data, material, sim_bf, sim_af, rmse_bf, rmse_af = test_csv(path_source, path_target)
    print(f"预测物质：{material}")
    print(f"相似度 - 迁移前: {sim_bf:.4f}，迁移后: {sim_af:.4f}")
    print(f"峰位RMSE - 迁移前: {rmse_bf:.4f}，迁移后: {rmse_af:.4f}")
