# losses.py
# 损失函数集合 
import math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks, savgol_filter, correlate
from scipy.optimize import minimize_scalar, linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.ndimage as ndi

# ---------- 更鲁棒的标量转 int ----------
def _to_int(x):
    if isinstance(x, int): return x
    if isinstance(x, float): return int(x)
    if torch.is_tensor(x): return int(x.item())
    if isinstance(x, np.ndarray): return int(x.item())
    return int(x)

# ========================== 基础工具函数 ==========================
def peak_aware_curvature_residual(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: float = 0.07,
    lam: float = 1.0,
):
    """
    Peak-aware curvature residual loss.
    用于抑制迁移过程中在非峰区捏造尖锐伪峰。

    Parameters
    ----------
    pred : Tensor, shape [B, L] or [B, 1, L]
        迁移后的光谱
    target : Tensor, shape [B, L] or [B, 1, L]
        目标光谱（显微拉曼）
    tau : float
        判断“目标谱是否为平坦区”的阈值（基于目标谱二阶差分）
    lam : float
        损失整体缩放因子

    Returns
    -------
    loss : Tensor
        标量损失
    """

    # 统一形状为 [B, L]
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    # 一阶差分
    d1_pred = pred[:, 1:] - pred[:, :-1]
    d1_tgt  = target[:, 1:] - target[:, :-1]

    # 二阶差分（曲率）
    d2_pred = d1_pred[:, 1:] - d1_pred[:, :-1]
    d2_tgt  = d1_tgt[:, 1:] - d1_tgt[:, :-1]

    # 目标谱的“平坦区掩码”
    # 只有在目标谱本身变化很小的位置，才严格惩罚 pred 的高曲率
    flat_mask = (torch.abs(d2_tgt) < tau).float()

    # 曲率残差（L1 比 L2 更稳，不会过度平滑）
    curvature_residual = torch.abs(d2_pred)

    # 只在目标谱平坦区惩罚 pred 的高曲率
    loss = (flat_mask * curvature_residual).mean()

    return lam * loss

def extract_segments_aligned(x_pred, x_tgt, M=300, O=150):
    """
    x_pred, x_tgt: [B, L] (float32)
    返回: pred_chunks, tgt_chunks: [B, K, M]
    """
    assert x_pred.dim() == 2 and x_tgt.dim() == 2
    B, L = x_pred.shape
    assert x_tgt.shape == (B, L)

    step, pad = M - O, M // 2
    device, dtype = x_pred.device, x_pred.dtype

    # 反射填充（与前向 _segment_forward 完全一致）
    x_pred_pad = F.pad(x_pred.unsqueeze(1), (pad, pad), mode='reflect')  # [B,1,L+2*pad]
    x_tgt_pad  = F.pad(x_tgt .unsqueeze(1), (pad, pad), mode='reflect')

    # unfold 切段 -> [B, K, 1, M] -> squeeze -> [B, K, M]
    pred_chunks = x_pred_pad.unfold(-1, M, step).squeeze(1)
    tgt_chunks  = x_tgt_pad .unfold(-1, M, step).squeeze(1)

    return pred_chunks, tgt_chunks
# ---------- 缝口导数平滑代理（拿不到 per-chunk 时用） ----------
def seam_smoothness_proxy(x, M=300, O=150, radius=5):
    """
    x: [B,L] 最终连续曲线
    在理论缝口处约束一阶导数一致性，替代 interface_continuity_loss
    """
    B, L = x.shape
    step = _to_int(M) - _to_int(O)
    seams = torch.arange(step, L, step, device=x.device)
    if seams.numel() == 0:
        return torch.tensor(0., device=x.device, dtype=x.dtype)
    dx = x[:, 1:] - x[:, :-1]  # [B,L-1]
    loss, cnt = 0.0, 0
    for s in seams.tolist():
        sL = max(1, s - radius)
        sR = min(L - 2, s + radius)
        dl = dx[:, sL-1:sL]     # 左边界导
        dr = dx[:, sR:sR+1]     # 右边界导
        loss = loss + (dl - dr).abs().mean()
        cnt += 1
    return loss / max(cnt, 1)

def smooth_segments_1d(x: torch.Tensor, k: int = 7) -> torch.Tensor:
    """
    对分段谱做轻微 1D 平滑（reflect pad + box filter）
    x: [B,K,M]
    return: [B,K,M]
    """
    if k is None or k <= 1:
        return x
    assert x.dim() == 3
    B, K, M = x.shape
    pad = k // 2

    # box filter
    w = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / k

    x2 = x.reshape(B * K, 1, M)              # [B*K,1,M]
    x2 = F.pad(x2, (pad, pad), mode='reflect')
    y2 = F.conv1d(x2, w)                     # [B*K,1,M]
    return y2.reshape(B, K, M)

# ---------- 一站式 Phase-A 段内损失（统一用 aligned-chunks） ----------
def compute_phase_a_losses(trans_A2B, domain_matched, raman_shift, M=300, O=150,
                           segment_chunks=None):
    """
    返回 dict:
      segment_peak / segment_sim / interface_continuity / segment_intensity / segment_intensity_ratio
    """
    # 1) 用与前向一致的 OLA 分段
    pred_chunks, tgt_chunks = extract_segments_aligned(trans_A2B, domain_matched, M=M, O=O)
    # ================= 真实拉曼轴分段（1:1 复刻 forward 的 reflect+unfold） =================
    if isinstance(raman_shift, np.ndarray):
        raman_shift = torch.from_numpy(raman_shift)

    raman_shift = raman_shift.to(trans_A2B.device).float()  # [L]

    B, K_pred, M_pred = pred_chunks.shape
    assert M_pred == M, f"M mismatch: pred_chunks M={M_pred} vs M={M}"

    step = M - O
    pad = M // 2

    # 和模型 forward 完全一致的 reflect padding
    raman_shift_pad = torch.nn.functional.pad(
        raman_shift.unsqueeze(0).unsqueeze(0),  # [1,1,L]
        (pad, pad),
        mode='reflect'
    ).squeeze(0).squeeze(0)  # [L+2*pad]

    # 在 padded 轴上 unfold
    raman_shift_segments = raman_shift_pad.unfold(0, M, step)  # [K_axis, M]

    # 对齐 pred_chunks 的 K（一般会刚好相等）
    K_axis = raman_shift_segments.shape[0]
    if K_axis != K_pred:
        raise RuntimeError(
            f"[AxisAlign] K mismatch after reflect-pad unfold: "
            f"axis K={K_axis}, pred K={K_pred}. "
            f"Check extract_segments_aligned consistency."
        )

    
    pred_for_peak = smooth_segments_1d(pred_chunks, k=7)
    tgt_for_peak = smooth_segments_1d(tgt_chunks, k=7)

    loss_segment_peak = segment_peak_alignment_loss(
        pred_for_peak,
        tgt_for_peak,
        raman_shift_segments=raman_shift_segments,
        tau=0.05,
        w_emd=1.0,
        w_simse=0.2,
        normalize_physical_scale=True
    )
    # 3) 段内相似度（zcorr + 导数）
    loss_segment_sim = segment_similarity_loss_zcorr(pred_chunks, tgt_chunks, alpha_grad=0.6)

    # 4) 段内强度（L2 + Hann + 弱段屏蔽；log-ratio + 裁剪 + 弱段屏蔽）
    loss_intensity       = segment_intensity_l2_loss(pred_chunks, tgt_chunks)
    loss_intensity_ratio = segment_intensity_ratio_loss(pred_chunks, tgt_chunks)

    # 5) 段间连续性：
    #    - 有模型 per-chunk（forward 暴露的 segment_chunks）就用它（真·缝口）
    #    - 否则用缝口导数平滑代理
    if segment_chunks is not None:
        loss_interface = interface_continuity_loss(segment_chunks, overlap=_to_int(O), p=1)
    else:
        loss_interface = seam_smoothness_proxy(trans_A2B, M=_to_int(M), O=_to_int(O), radius=5)

    return {
        'segment_peak'            : loss_segment_peak,
        'segment_sim'             : loss_segment_sim,
        'interface_continuity'    : loss_interface,
        'segment_intensity'       : loss_intensity,
        'segment_intensity_ratio' : loss_intensity_ratio,
    }

# ---------------------------
# 段内 L2（带 Hann 窗 & 弱段屏蔽）
# ---------------------------
def segment_intensity_l2_loss(pred_chunks, target_chunks, window=None, thr_q=0.20):
    """
    pred_chunks, target_chunks: [B, K, M]
    window: None 或 [M]，不传则自动用 Hann(M)
    thr_q: 按 target 段均值的分位数屏蔽弱段，默认屏蔽最弱的 15%
    """
    assert pred_chunks.shape == target_chunks.shape
    B, K, M = pred_chunks.shape
    device, dtype = pred_chunks.device, pred_chunks.dtype

    if window is None:
        w = torch.hann_window(M, periodic=False, device=device, dtype=dtype)  # 与 OLA 匹配
    else:
        w = window.to(device=device, dtype=dtype)
        assert w.numel() == M
    w = w.view(1, 1, M)

    diff2 = (pred_chunks - target_chunks).pow(2) * w  # [B,K,M]

    # 弱段屏蔽：target 段均值极小的窗口不计入（避免初期/外对齐导致的 0 区域爆损）
    mean_t = target_chunks.abs().mean(dim=-1)            # [B,K]
    thr = torch.quantile(mean_t, thr_q, dim=1, keepdim=True)  # 每个样本自己的阈值
    mask = (mean_t >= thr).float().unsqueeze(-1)         # [B,K,1]

    diff2 = diff2 * mask
    per_seg = diff2.mean(dim=-1)                         # [B,K]
    loss = per_seg.sum() / (mask.squeeze(-1).sum() + 1e-6)
    return loss


# ---------------------------
# 段内强度比例（对数域 + 裁剪 + 弱段屏蔽）—推荐
# ---------------------------
def segment_intensity_ratio_loss(pred_chunks, target_chunks, eps=1e-6,
                                 clip=(0.25, 4.0), thr_q=0.20):
    """
    让每段 mean(|pred|)/mean(|target|) → 1
    - 用 log 域平方，更对称稳健（×2 和 ×0.5 惩罚相同）
    - 对比例裁剪，防极端值爆梯度
    - 屏蔽 target 极弱的段，避免分母≈0 造成震荡
    """
    assert pred_chunks.shape == target_chunks.shape
    mean_p = pred_chunks.abs().mean(dim=-1) + eps   # [B,K]
    mean_t = target_chunks.abs().mean(dim=-1) + eps

    # 弱段屏蔽
    thr  = torch.quantile(mean_t, thr_q, dim=1, keepdim=True)
    mask = (mean_t >= thr).float()                  # [B,K]

    ratio = (mean_p / mean_t).clamp(min=clip[0], max=clip[1])  # [B,K]
    loss  = (torch.log(ratio) ** 2) * mask
    loss  = loss.sum() / (mask.sum() + 1e-6)
    return loss

def _gaussian_kernel1d(sigma, device, dtype):
    """生成1D高斯核"""
    radius = int(max(1, round(3*sigma)))
    x = torch.arange(-radius, radius+1, device=device, dtype=dtype)
    k = torch.exp(-0.5*(x/sigma)**2)
    k = k / (k.sum() + 1e-8)
    return k.view(1,1,-1), radius

def _conv1d_reflect(x, k, radius):
    """使用反射填充的1D卷积"""
    xpad = F.pad(x, (radius, radius), mode='reflect')
    return F.conv1d(xpad, k)

def _rolling_median_1d(x, window):
    """1D滚动中值滤波"""
    B, K, L = x.shape
    out = x.clone()
    half = window // 2
    
    for i in range(L):
        start = max(0, i - half)
        end = min(L, i + half + 1)
        out[:, :, i] = x[:, :, start:end].median(dim=-1)[0]
    
    return out

def _physical_preprocess(y, baseline_window=15, log_compress=True, z_score=True, smooth_sigma=1.5):
    """物理域预处理：去基线、log压缩、标准化、平滑"""
    # 1) 滚动中值去基线
    if baseline_window > 0:
        y = y - _rolling_median_1d(y, baseline_window)
    
    # 2) log1p压缩（抗动态范围）
    if log_compress:
        y = torch.log1p(torch.clamp(y, min=0.0))
    
    # 3) Z-score标准化（抗幅度差）
    if z_score:
        y = (y - y.mean(dim=-1, keepdim=True)) / (y.std(dim=-1, keepdim=True) + 1e-6)
    
    # 4) 窄高斯平滑（抗噪声）
    if smooth_sigma > 0:
        gk, r = _gaussian_kernel1d(smooth_sigma, y.device, y.dtype)
        y = _conv1d_reflect(y.view(-1, 1, y.shape[-1]), gk, r).view_as(y)
    
    return y

def _log_filter_1d(x, sigma=1.0):
    """LoG滤波器：检测峰位"""
    gk, r = _gaussian_kernel1d(sigma, x.device, x.dtype)
    # 二阶导数核
    lap = torch.tensor([1., -2., 1.], device=x.device, dtype=x.dtype).view(1,1,3)
    x_smooth = _conv1d_reflect(x.view(-1, 1, x.shape[-1]), gk, r)
    log_resp = _conv1d_reflect(x_smooth, lap, 1)
    return torch.relu(-log_resp).view_as(x)  # 负LoG取正，突出峰位

def _soft_peak_map_physical(y, beta=5.0, baseline_window=15, log_compress=True, z_score=True, smooth_sigma=1.5):
    """物理域的软峰图生成"""
    # 预处理
    y_proc = _physical_preprocess(y, baseline_window, log_compress, z_score, smooth_sigma)
    
    # LoG峰检测
    peak_logits = _log_filter_1d(y_proc, sigma=1.0)
    
    # 温度缩放的softmax
    P = torch.softmax(beta * peak_logits, dim=-1)
    return P

def _dice_jaccard_loss(P_pred, P_tgt, mode='dice', eps=1e-6):
    """Dice/Jaccard损失"""
    if mode == 'dice':
        intersection = (P_pred * P_tgt).sum(dim=-1)
        union = P_pred.sum(dim=-1) + P_tgt.sum(dim=-1)
        dice = (2.0 * intersection) / (union + eps)
        return 1.0 - dice.mean()
    else:  # jaccard
        intersection = (P_pred * P_tgt).sum(dim=-1)
        union = P_pred.sum(dim=-1) + P_tgt.sum(dim=-1) - intersection
        jaccard = intersection / (union + eps)
        return 1.0 - jaccard.mean()

def _overlap_indices(L, O, trim_ratio=0.2, device=None):
    """获取重叠区域的有效索引"""
    cut = int(O * trim_ratio)
    tail_idx = torch.arange(L - O + cut, L - cut, device=device)
    head_idx = torch.arange(cut, O - cut, device=device)
    return tail_idx, head_idx

def _emd_1d(P, Q):
    """1D Earth Mover's Distance"""
    Pc = P.cumsum(dim=-1)
    Qc = Q.cumsum(dim=-1)
    return (Pc - Qc).abs().mean(dim=-1)

def smooth1d_reflect(x, ksize=5):
    """使用反射填充的1D平滑"""
    k = torch.ones(1,1,ksize, device=x.device, dtype=x.dtype) / ksize
    r = ksize // 2
    xpad = F.pad(x, (r, r), mode='reflect')
    return F.conv1d(xpad, k)

# ========================== 核心损失函数 ==========================

def _simse(pred, target, eps=1e-8):
    """
    Scale-Invariant MSE: min_s || s*pred - target ||^2
    闭式解 s* = (pred·target)/(pred·pred)
    """
    num = (pred * target).sum(dim=-1, keepdim=True)
    den = (pred * pred).sum(dim=-1, keepdim=True) + eps
    s = num / den
    err = (s * pred - target) ** 2
    return err.mean()

def _make_log_kernel_1d(sigma: float, dtype, device):
    # 离散 1D LoG 核，长度取 6σ 向上取奇数
    half = int(math.ceil(3.0 * sigma))
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    a = (x ** 2 - sigma ** 2) / (sigma ** 4)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    log = a * g
    # 零均值
    log = log - log.mean()
    # 形状为 [1,1,K] 以便 conv1d
    return log.view(1, 1, -1)

def _multiscale_log_response(x, sigmas=(1.0, 2.0, 3.0), thresh_k=2.5):
    """
    x: [B, L]  -> 返回非负响应 [B, L]（多尺度最大值 + 鲁棒门限）
    """
    B, L = x.shape
    device, dtype = x.device, x.dtype
    x1 = x.unsqueeze(1)  # [B,1,L]
    resps = []
    for s in sigmas:
        k = _make_log_kernel_1d(float(s), dtype, device)
        r = F.conv1d(x1, k, padding=k.size(-1)//2).squeeze(1)  # [B,L]
        resps.append(r)
    r = torch.stack(resps, dim=0).amax(dim=0)  # 多尺度取最大
    r = F.relu(r)

    # 鲁棒门限：median + k * MAD
    med = r.median(dim=-1, keepdim=True).values
    mad = (r - med).abs().median(dim=-1, keepdim=True).values + 1e-8
    r = torch.where(r > med + thresh_k * mad, r, torch.zeros_like(r))
    return r

def _sinkhorn_ot_cost(mu, nu, C, eps=0.1, iters=60):
    """
    可微 Sinkhorn-Knopp 计算批量 OT 代价
    mu, nu: [B, N], [B, M]  (均为正且和相等)
    C: [N, M]  代价矩阵
    返回: 标量 batch 平均 cost
    """
    B, N = mu.shape
    M = nu.shape[1]
    K = torch.exp(-C / eps).to(mu.device)      # [N,M]
    K = K.clamp_min(1e-12)

    u = torch.ones(B, N, device=mu.device, dtype=mu.dtype) / N
    v = torch.ones(B, M, device=mu.device, dtype=mu.dtype) / M

    K_t = K.transpose(0, 1)                    # [M,N]，供转置乘法使用

    for _ in range(iters):
        Kv = torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + 1e-12  # [B,N]
        u = mu / Kv
        KTu = torch.matmul(K_t, u.unsqueeze(-1)).squeeze(-1) + 1e-12  # [B,M]
        v = nu / KTu

    # 运输计划 Π = diag(u) K diag(v)
    # 代价 = sum Π ⊙ C
    # 为了省显存，分两步乘法
    Ku = (u.unsqueeze(-1) * K)                 # [B,N,M]
    P = Ku * v.unsqueeze(1)                    # [B,N,M]
    cost = (P * C) .sum(dim=(1, 2))            # [B]
    return cost.mean()

def peak_alignment_loss_soft(
    pred, target, raman_shift,          # 与原接口保持一致
    tau=0.08,
    # 峰检测
    sigmas=(1.0, 2.0, 3.0),
    thresh_k=2.0,
    # OT 约束（全部在 bins 空间）
    radius_bins=25,                     # 最大可匹配半径（bins）
    lam=1.1,                            # 尘盒成本（归一化后，略高于1）
    sink_mass=1e-3,                     # 尘盒初始质量占比
    # Sinkhorn
    sinkhorn_eps=0.1,
    sinkhorn_iters=60,
    # 组合权重
    w_emd=1.0,
    w_simse=0.2
):
    """
    软峰-EMD（索引距离归一化 + 半径截断 + 尘盒）+ 全谱 SIMSE
    pred/target: [B,L]；raman_shift 不参与代价，仅作接口兼容
    返回: 标量损失
    """
    device, dtype = pred.device, pred.dtype
    B, L = pred.shape

    # 1) 软峰概率：多尺度 LoG + 门限 + softmax/tau
    resp_p = _multiscale_log_response(pred,   sigmas=sigmas, thresh_k=thresh_k) + 1e-12
    resp_t = _multiscale_log_response(target, sigmas=sigmas, thresh_k=thresh_k) + 1e-12
    P_src = F.softmax(resp_p / tau, dim=-1)   # [B,L]
    P_tgt = F.softmax(resp_t / tau, dim=-1)   # [B,L]

    # 2) 代价矩阵：用“索引距离（bins）”，半径截断并归一化到 [0,1]
    idx = torch.arange(L, device=device, dtype=dtype)
    C = (idx[None, :] - idx[:, None]).abs()            # [L,L]
    BIG = torch.tensor(1e6, device=device, dtype=dtype)
    C = torch.where(C > int(radius_bins), BIG, C)      # 超半径不可达
    C = C / float(max(1, int(radius_bins)))            # 归一化到 [0,1]

    # 3) 扩展尘盒：最后一行/列代价=lam，尘盒<->尘盒为0
    C_ext = torch.full((L + 1, L + 1), float(lam), device=device, dtype=dtype)
    C_ext[:L, :L] = C
    C_ext[L, L] = 0.0

    # 4) 分布扩到 L+1，加入少量尘盒质量，保证和一致
    sink = torch.full((B, 1), float(sink_mass), device=device, dtype=dtype)
    mu = torch.cat([P_src, sink], dim=-1)
    nu = torch.cat([P_tgt, sink], dim=-1)
    mu = mu / (mu.sum(dim=-1, keepdim=True) + 1e-12)
    nu = nu / (nu.sum(dim=-1, keepdim=True) + 1e-12)

    # 5) Sinkhorn-EMD（注意现在代价规模 ~O(1)）
    emd_cost = _sinkhorn_ot_cost(mu, nu, C_ext, eps=sinkhorn_eps, iters=sinkhorn_iters)

    # 6) 全谱 SIMSE（形状锚点）
    simse_cost = _simse(pred, target)

    loss = w_emd * emd_cost + w_simse * simse_cost

    return loss


def segment_peak_alignment_loss_physical(segments_pred, segments_tgt, 
                                        beta=5.0, baseline_window=15, 
                                        log_compress=True, z_score=True, smooth_sigma=1.5,
                                        segment_weights=None):
    """物理强度域的段内峰位对齐损失"""
    B, K, M = segments_pred.shape
    
    # 生成软峰图
    P_pred = _soft_peak_map_physical(segments_pred, beta, baseline_window, log_compress, z_score, smooth_sigma)
    P_tgt  = _soft_peak_map_physical(segments_tgt,  beta, baseline_window, log_compress, z_score, smooth_sigma)
    
    # Dice损失
    loss = _dice_jaccard_loss(P_pred, P_tgt, mode='dice')
    
    return loss

def segment_similarity_loss_zcorr(segments_pred, segments_tgt, alpha_grad=0.3, eps=1e-8):
    """
    段内相似度：z-corr（值相关 + 梯度相关）
    返回的是一个标准化的 loss ∈ [0, 2] 附近（1 - 相关），不再出现巨大正负。
    """
    B, K, M = segments_pred.shape

    # 段内标准化
    pred_std = (segments_pred - segments_pred.mean(dim=-1, keepdim=True)) / (segments_pred.std(dim=-1, keepdim=True) + eps)
    tgt_std  = (segments_tgt  - segments_tgt .mean(dim=-1, keepdim=True)) / (segments_tgt .std(dim=-1, keepdim=True) + eps)

    # 值相关：用 mean 不是 sum
    val_corr = (pred_std * tgt_std).mean(dim=-1)  # [B,K], 约在 [-1,1]

    # 梯度相关（可选，仍然 mean）
    if alpha_grad > 0:
        pred_g = torch.diff(pred_std, dim=-1)
        tgt_g  = torch.diff(tgt_std,  dim=-1)
        grad_corr = (pred_g * tgt_g).mean(dim=-1)  # [B,K], 约在 [-1,1]
        corr = val_corr + alpha_grad * grad_corr
    else:
        corr = val_corr

    # 裁剪，防数值抖动
    corr = torch.clamp(corr, -1.0, 1.0)

    # "越小越相似"的 loss 形式
    return (1.0 - corr).mean()

def seam_penalty_from_segments(seg_win, window, overlap, value_weight=1.0, slope_weight=1.0, eps=1e-4):
    """段间连续性损失：值连续 + 斜率连续"""
    B, S, L = seg_win.shape
    device = seg_win.device
    w = window.to(device)

    tail_idx, head_idx = _overlap_indices(L, overlap, device=device)
    # 去窗（并屏蔽 w 很小的位置）
    wL = w[tail_idx].clamp_min(eps)   # (Oo,)
    wR = w[head_idx].clamp_min(eps)   # (Oo,)

    # 权重：两窗越大处权重越高，避免边缘噪声
    weight = (wL * wR).sqrt()[None, None, :]   # (1,1,Oo)

    seam_vals = []
    seam_slopes = []

    for s in range(S - 1):
        left_tail  = seg_win[:, s,   :][:, tail_idx] / wL   # (B, Oo)
        right_head = seg_win[:, s+1, :][:, head_idx] / wR   # (B, Oo)

        # 值连续性
        v_loss = (weight * (left_tail - right_head).abs().unsqueeze(1)).mean()

        # 斜率连续性（对重叠区内部做一阶差分后再对齐）
        dL = left_tail[:, 1:]  - left_tail[:, :-1]          # (B, Oo-1)
        dR = right_head[:, 1:] - right_head[:, :-1]         # (B, Oo-1)
        w_s = weight[..., 1:]                               # (1,1,Oo-1)
        s_loss = (w_s * (dL - dR).abs().unsqueeze(1)).mean()

        seam_vals.append(v_loss)
        seam_slopes.append(s_loss)

    seam_val   = torch.stack(seam_vals).mean()   * value_weight
    seam_slope = torch.stack(seam_slopes).mean() * slope_weight
    return seam_val, seam_slope



def masked_cosine_similarity(a, b, M=None, eps=1e-8):
    # a,b: [B,N] 或 [B,1,N]
    a = a.squeeze(1) if a.dim()==3 else a
    b = b.squeeze(1) if b.dim()==3 else b
    if M is not None:
        # M 可以是掩码 tensor 或整数索引
        if torch.is_tensor(M):
            if M.dim() == 1 and M.numel() == 1:
                # 单个元素的 tensor，转换为整数索引
                M = M.item()
                a = a[..., :M]; b = b[..., :M]
            else:
                # 掩码 tensor，直接应用
                a = a * M
                b = b * M
        else:
            # 整数索引
            a = a[..., :M]; b = b[..., :M]
    num = (a*b).sum(dim=-1)
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    cos = num / (na*nb + eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    # 返回"loss"语义：1 - cos（越小越相似）
    return (1.0 - cos).mean()

def gradient_structure_loss(pred, target):
    """梯度结构损失：匹配光谱的一阶导数结构"""
    pred_grad = torch.diff(pred, dim=-1)
    target_grad = torch.diff(target, dim=-1)
    
    # 计算梯度的MSE
    grad_loss = F.mse_loss(pred_grad, target_grad)
    
    return grad_loss

def intensity_matching_loss(pred, target, high_freq_weight=2.0):
    """强度匹配损失，对高频区域给予更高权重"""
    # 计算强度差异
    intensity_diff = pred - target
    
    # 构造频率权重：高频区域权重更高
    L = pred.shape[-1]
    freq_weights = torch.linspace(1.0, high_freq_weight, L, device=pred.device)
    freq_weights = freq_weights.view(1, 1, -1).expand_as(pred)
    
    # 加权MSE损失
    weighted_loss = freq_weights * intensity_diff ** 2
    return weighted_loss.mean()

def spectral_correlation_loss(pred, target, window_size=100):
    """局部光谱相关性损失，使用滑动窗口计算局部相关性"""
    batch_size, seq_len = pred.shape
    device = pred.device
    
    total_corr = 0.0
    valid_windows = 0
    
    for i in range(batch_size):
        pred_spectrum = pred[i]
        target_spectrum = target[i]
        
        # 滑动窗口计算局部相关性
        for start in range(0, seq_len - window_size + 1, window_size // 2):
            end = start + window_size
            
            pred_window = pred_spectrum[start:end]
            target_window = target_spectrum[start:end]
            
            # 计算相关系数
            pred_centered = pred_window - pred_window.mean()
            target_centered = target_window - target_window.mean()
            
            numerator = (pred_centered * target_centered).sum()
            # 安全计算分母，避免负数开方
            pred_var = (pred_centered ** 2).sum()
            target_var = (target_centered ** 2).sum()
            denominator = torch.sqrt(torch.clamp(pred_var * target_var, min=1e-16)) + 1e-8
            
            corr = numerator / denominator
            total_corr += corr
            valid_windows += 1
    
    # 返回负相关，避免除零
    if valid_windows == 0:
        return torch.tensor(0.0, device=device)
    return 1.0 - total_corr / valid_windows

def l1_loss(pred, target, mask=None, high_freq_weight=1.5):
    """掩码L1损失，支持高频权重"""
    if mask is not None:
        diff = (pred - target) * mask
        loss = diff.abs().sum() / (mask.sum() + 1e-8)
    else:
        # 高频权重
        L = pred.shape[-1]
        freq_weights = torch.linspace(1.0, high_freq_weight, L, device=pred.device)
        freq_weights = freq_weights.view(1, 1, -1).expand_as(pred)
        diff = (pred - target) * freq_weights
        loss = diff.abs().mean()
    
    return loss

# ========================== 当前使用的损失函数 ==========================

def _as_BN(x):
    """辅助函数：把 [B,1,N]/[B,N] 统一成 [B,N]"""
    if x.dim() == 3:  # [B,1,N]
        return x.squeeze(1)
    return x  # [B,N]

def _soft_peak_map_simple(sig_BN: torch.Tensor, beta: float = 8.0) -> torch.Tensor:
    # sig_BN: [B,N]
    global _log_kernel
    x = sig_BN.unsqueeze(1)
    if _log_kernel is None or _log_kernel.device != x.device:
        _log_kernel = torch.tensor([1., -2., 1.], device=x.device).view(1,1,3)
    s = F.conv1d(x, _log_kernel, padding=1).squeeze(1)   # [B,N]
    s = F.relu(s)
    # 若整条为0，给一个极小常量，避免 softmax 全 0
    zero_mask = (s.sum(dim=-1, keepdim=True) <= 1e-12)
    s = s + zero_mask * 1e-6
    P = torch.softmax(beta * s, dim=-1)
    return P

_log_kernel = None  # 全局缓存

def peak_emd_loss(pred, target, x, tau: float = 0.10, beta: float = None):
    pred = pred.squeeze(1) if pred.dim()==3 else pred  # [B,N]
    target = target.squeeze(1) if target.dim()==3 else target
    if beta is None: beta = 1.0 / max(tau, 1e-6)

    Pp = _soft_peak_map_simple(pred,   beta=beta)  # [B,N]
    Pt = _soft_peak_map_simple(target, beta=beta)

    x = x.to(pred.device)
    # 归一化坐标，数值与光谱点数相关，而不再直接受 cm^-1 跨度影响
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    # dx = torch.diff(x, prepend=x[:1])
    dx = torch.diff(x, dim=-1, prepend=x[..., :1])

    # dx = torch.diff(x, prepend=x[:1]).to(pred.device)
    # 测度修正 + 安全归一化：sum<eps 时退回均匀分布
    Pp = Pp * dx; Pt = Pt * dx
    sumP = Pp.sum(dim=-1, keepdim=True)
    sumT = Pt.sum(dim=-1, keepdim=True)
    eps = 1e-12
    uni = torch.full_like(Pp, 1.0 / Pp.size(-1))
    Pp = torch.where(sumP > eps, Pp / (sumP + eps), uni)
    Pt = torch.where(sumT > eps, Pt / (sumT + eps), uni)

    cdfP = torch.cumsum(Pp, dim=-1)
    cdfQ = torch.cumsum(Pt, dim=-1)
    emd  = torch.mean(torch.sum((cdfP - cdfQ).abs() * dx, dim=-1))
    return emd

def simse_loss(pred, target, eps=1e-8):
    pred = pred.squeeze(1) if pred.dim()==3 else pred
    target = target.squeeze(1) if target.dim()==3 else target
    num = (pred*target).sum(dim=-1)
    den = (pred*pred).sum(dim=-1) + eps
    a = (num/den).unsqueeze(-1)
    resid = a*pred - target
    return (resid*resid).mean()

def emd_simse_loss(pred, target, x, tau=0.10, w_emd=1.0, w_simse=0.5):
    """
    组合：EMD + SIMSE
    推荐作为"峰位+强度差"的主损组合：
    - EMD：拉齐峰位（可容忍大位移，多峰友好）
    - SIMSE：允许整体缩放，突出形状/局部强度差异
    """
    L_emd = peak_emd_loss(pred, target, x, tau=tau)
    L_sim = simse_loss(pred, target)
    return w_emd * L_emd + w_simse * L_sim, {"emd": L_emd.detach(), "simse": L_sim.detach()}

def _lin_sched(a, b, t0, t1, t):
    """线性调度函数"""
    if t <= t0: return a
    if t >= t1: return b
    r = (t - t0) / max(t1 - t0, 1e-6)
    return a + r * (b - a)

def global_similarity_loss_emd_simse(pred, target, raman_shift, epoch,
                                     phase_start=50, phase_end=120,
                                     tau_start=0.2, tau_end=0.08,
                                     w_emd=1.0, w_simse=0.4):
    """
    45轮后使用：全谱 EMD为主 + SIMSE辅助
    - τ 从 tau_start 线性降到 tau_end（先软后尖）
    - 对幅度差异不敏感（SIMSE自动拟合尺度a），强调形状/峰分布
    """
    # 线性退火 τ（在 [phase_start, phase_end] 段内生效，段外钳制端点）
    tau = _lin_sched(tau_start, tau_end, phase_start, phase_end, epoch)

    # 统一 device / dtype
    if not torch.is_tensor(raman_shift):
        x = torch.tensor(raman_shift, device=pred.device, dtype=pred.dtype)
    else:
        x = raman_shift.to(device=pred.device, dtype=pred.dtype)

    loss, parts = emd_simse_loss(pred, target, x, tau=tau, w_emd=w_emd, w_simse=w_simse)
    # 附带诊断信息可选返回（需要的话在外层log）
    return loss, parts, tau

def interface_continuity_loss(seg_chunks_pred, overlap: int = 50, p: int = 2):
    """
    段间连续性损失函数
    seg_chunks_pred: [B, K, M]，分段输出（未 OLA 拼接）  
    overlap: 接口重叠的点数 O  
    p: 用于差异范数（常用 p=2 或 p=1）

    思路：对每一对相邻段 i, i+1，
      - 取左段的后 overlap 点
      - 取右段的前 overlap 点
      - 计算它们在重叠区域的差异（高度 /一阶差分）作为损失

    返回：标量损失
    """
    B, K, M = seg_chunks_pred.shape
    if K < 2:
        return torch.tensor(0.0, device=seg_chunks_pred.device, requires_grad=True)

    loss_sum = 0.0
    count = 0
    for i in range(K - 1):
        left = seg_chunks_pred[:, i, M - overlap : M]    # [B, overlap]
        right = seg_chunks_pred[:, i+1, : overlap]        # [B, overlap]
        diff = left - right                               # [B, overlap]

        # 高度连续性惩罚：L_p
        loss_sum = loss_sum + torch.mean(torch.abs(diff) ** p)

        # 可选：斜率 / 一阶连续性惩罚
        if overlap >= 2:
            diff_left_slope = left[:, 1:] - left[:, :-1]
            diff_right_slope = right[:, 1:] - right[:, :-1]
            slope_diff = diff_left_slope - diff_right_slope
            loss_sum = loss_sum + torch.mean(torch.abs(slope_diff) ** p)

        count += 1

    return loss_sum / count

def _gaussian_kernel1d_phase_a(sigma, device, dtype):
    """生成一维高斯核（Phase A版本）"""
    radius = int(3*sigma)
    x = torch.arange(-radius, radius+1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma)**2)
    return (k / (k.sum() + 1e-8)).view(1,1,-1), radius

def _soft_peak_map(y, gauss_sigma=2.0, tau=0.06):
    """生成软峰概率图"""
    # y: [..., M]
    *head, M = y.shape
    x = y.reshape(-1, 1, M)                 # [N,1,M]
    gk, r = _gaussian_kernel1d_phase_a(gauss_sigma, y.device, y.dtype)
    # 使用正确的padding来保持维度
    kernel_size = gk.shape[-1]
    padding = (kernel_size - 1) // 2
    y_s = F.conv1d(x, gk, padding=padding)   # 保持维度
    lap = torch.tensor([-1., 2., -1.], device=y.device, dtype=y.dtype).view(1,1,3)
    resp = torch.relu(-F.conv1d(y_s, lap, padding=1)).squeeze(1)  # [N,M]
    P = torch.softmax(resp / tau, dim=-1)
    result = P.view(*head, M)                 # [..., M]
    return result


def segment_peak_alignment_loss(
    segments_pred, segments_tgt,
    raman_shift_segments,             
    segment_weights=None,
    tau=0.05,
    w_emd=1.0,
    w_simse=0.2,
    normalize_physical_scale=True
):
    """
    段内峰位对齐损失（基于真实拉曼轴，cm^-1）
    segments_pred, segments_tgt: [B, K, M]
    raman_shift_segments:        [B, K, M] or [K, M]
    """
    assert segments_pred.dim() == 3
    B, K, M = segments_pred.shape
    device, dtype = segments_pred.device, segments_pred.dtype

    # ---------- 1. 处理拉曼轴 ----------
    if raman_shift_segments.dim() == 2:
        # [K, M] -> [B, K, M]
        raman_shift_segments = (
            raman_shift_segments
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

    raman_shift_segments = raman_shift_segments.to(device=device, dtype=dtype)

    # ---------- 2. 可选：物理尺度归一化（强烈推荐） ----------
    if normalize_physical_scale:
        # 每一段用自己的物理跨度做归一化，避免低波数吃亏
        seg_min = raman_shift_segments.min(dim=-1, keepdim=True).values
        seg_max = raman_shift_segments.max(dim=-1, keepdim=True).values
        raman_norm = (raman_shift_segments - seg_min) / (seg_max - seg_min + 1e-8)
    else:
        raman_norm = raman_shift_segments

    # ---------- 3. 逐段计算 EMD + SIMSE ----------
    segment_losses = []

    for k in range(K):
        pred_seg = segments_pred[:, k, :]          # [B, M]
        tgt_seg  = segments_tgt[:, k, :]           # [B, M]
        x_seg    = raman_norm[:, k, :]              # [B, M]

        seg_loss, _ = emd_simse_loss(
            pred=pred_seg,
            target=tgt_seg,
            x=x_seg,
            tau=tau,
            w_emd=w_emd,
            w_simse=w_simse
        )
        segment_losses.append(seg_loss)             # scalar (mean over B)

    # ---------- 4. 段权重 ----------
    if segment_weights is None:
        with torch.no_grad():
            Pt = _soft_peak_map_physical(
                segments_tgt,
                beta=1.0 / max(tau, 1e-6)
            )                                       # [B,K,M]
            w_seg = Pt.sum(dim=-1)                  # [B,K]
            w_seg = w_seg + 0.01  # 或者 0.02加权重保持，可删除
            w_seg = w_seg / (w_seg.sum(dim=-1, keepdim=True) + 1e-8)
    else:
        w_seg = segment_weights
        if w_seg.dim() == 1:
            w_seg = w_seg.unsqueeze(0).expand(B, -1)
        w_seg = w_seg + 0.01  # 或者 0.02加权重保持，可删除
        w_seg = w_seg / (w_seg.sum(dim=-1, keepdim=True) + 1e-8)

    # ---------- 5. 加权汇总 ----------
    total_loss = 0.0
    for k, seg_loss in enumerate(segment_losses):
        total_loss += seg_loss * w_seg[:, k].mean()

    return total_loss / K


