from scipy.signal import savgol_filter, find_peaks
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# ----------------------------
# 
# ----------------------------
def conv1x(kernel_size):
    pad = (kernel_size - 1) // 2
    return nn.Conv1d(1, 1, kernel_size, padding=pad, bias=False)

def convC(inC, outC, k=3, d=1):
    pad = d * (k - 1) // 2
    return nn.Conv1d(inC, outC, k, padding=pad, dilation=d)

def depthwise_conv(c, k=3, d=1):
    pad = d * (k - 1) // 2
    return nn.Conv1d(c, c, k, padding=pad, dilation=d, groups=c)
# ----------------------------
# 多尺度 + SE 门控
# ----------------------------
class SE1D(nn.Module):
    def __init__(self, C, r=8):
        super().__init__()
        self.fc1 = nn.Conv1d(C, C // r, 1)
        self.fc2 = nn.Conv1d(C // r, C, 1)

        self.se_sparsity_loss = 0.0
        self.branch_corr_penalty = 0.0

    def forward(self, x):
        s = F.adaptive_avg_pool1d(x, 1)
        a = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        # sparsity（鼓励更尖锐的注意力分配）
        self.se_sparsity_loss = (a.mean() - 0.5).abs()
        return x * a, a


class MultiScaleGate1D(nn.Module):
    def __init__(self, inC=1, midC=24, k_list=(3, 9, 15), d_list=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([convC(inC, midC, k, d) for k, d in zip(k_list, d_list)])
        self.se = SE1D(midC * len(k_list), r=6)
        self.proj = nn.Conv1d(midC * len(k_list), 1, 1)

    def forward(self, x, return_vis=False):
        # 分支 & 拼接
        feats = [b(x) for b in self.branches]             # [B,midC,L] * nb
        z = torch.cat(feats, dim=1)                       # [B,midC*nb,L]
        z_se, a = self.se(z)                              # [B,midC*nb,L], [B,midC*nb,1]
        y = self.proj(z_se) + x                           # 残差
        if not return_vis:
            return y
        # 可视化信息
        with torch.no_grad():
            nb = len(self.branches)
            B, C, L = z.shape
            mid = C // nb
            # 每个分支的平均响应
            branch_mean = torch.stack([z[:, i * mid:(i + 1) * mid].mean(dim=1)
                                       for i in range(nb)], dim=1)  # [B,nb,L]
            gate_curve = torch.sigmoid(self.proj.weight.mean()) * torch.ones(B, 1, L, device=x.device)
        return y, dict(branch_mean=branch_mean, se_weights=a.squeeze(-1), gate_curve=gate_curve.squeeze(1))



class ImprovedXAxisWarp(nn.Module):
    """
    
      - 多尺度特征提取
     

    forward:
      x: [B,1,L]
      return: x_warped: [B,1,L], total_shift: [B,1,L]
    """

    def __init__(self, L=1500, max_shift_bins=20, K_ctrl=16, use_freq_weights=False):
        super().__init__()
        self.L = int(L)
        self.max_shift = float(max_shift_bins)
        self.K = int(K_ctrl)
        self.use_freq_weights = bool(use_freq_weights)

        # 多尺度卷积特征（与旧版保持风格一致）
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 8, 3, padding=1), nn.BatchNorm1d(8), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 8, 3, padding=1), nn.BatchNorm1d(8), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 8, 3, padding=1), nn.BatchNorm1d(8), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 8, 3, padding=1), nn.BatchNorm1d(8), nn.ReLU()
        )

        # 融合成 32通道，再做 1x1 精炼（输出不再直接给 local_shift）
        self.fusion = nn.Sequential(
            nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 32, 1), nn.BatchNorm1d(32), nn.ReLU()
        )

        # 控制点位移预测头：输出 K 个控制点位移（标量序列）
        self.ctrl_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(64),   # 稍微聚合，稳一点
            nn.Flatten(),               # 32*64
            nn.Linear(32 * 64, 128), nn.ReLU(),
            nn.Linear(128, self.K)      # [B,K]
        )

        # 全局位移（单标量，再广播）
        self.global_shift_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(32 * 64, 128), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)           # [B,1]
        )


        self.smooth_l1 = torch.tensor(0.0)
        self.smooth_l2 = torch.tensor(0.0)
        self.flip_penalty = torch.tensor(0.0)  # 位移方向折返惩罚



    def forward(self, x):
        # x: [B,1,L]
        B, C, L = x.shape
        assert C == 1 and L == self.L, "判断输入长度需与初始化一致"

        # 多尺度特征
        f1, f2, f3, f4 = self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)  # [B,8,Li]
        feats = [f1, f2, f3, f4]
        feats = [F.interpolate(f, size=L, mode='linear', align_corners=False) if f.size(-1) != L else f
                 for f in feats]
        feat_cat = torch.cat(feats, dim=1)                    # [B,32,L]
        feat_cat = self.fusion(feat_cat)                      # [B,32,L]

        # 1) 控制点位移（[-max_shift, max_shift]）
        ctrl = self.ctrl_head(feat_cat)                       # [B,K]
        ctrl = torch.tanh(ctrl) * self.max_shift             # [B,K]

        # 2) 线性插值到 L
        local_shift = F.interpolate(ctrl.unsqueeze(1), size=L, mode='linear', align_corners=True)  # [B,1,L]


        # 3) 全局位移（[-0.2*max_shift, 0.2*max_shift]），广播到 L
        g_shift = torch.tanh(self.global_shift_predictor(feat_cat)) * (self.max_shift * 0.1)  # [B,1]
        g_shift = g_shift.unsqueeze(-1).expand(-1, -1, L)                                      # [B,1,L]

        # 合成总位移并裁剪到范围
        total_shift = torch.clamp(local_shift + g_shift, -self.max_shift, self.max_shift)      # [B,1,L]

        # 4) 平滑正则（对 total_shift 的一阶/二阶差分）
        d1 = total_shift[:, :, 1:] - total_shift[:, :, :-1]
        d2 = d1[:, :, 1:] - d1[:, :, :-1]
        self.smooth_l1 = d1.abs().mean()
        self.smooth_l2 = d2.pow(2).mean()


        disp_all = total_shift                       # [B,1,L]
        d1 = disp_all[..., 1:] - disp_all[..., :-1]  # 一阶导
        # 相邻一阶导异号 -> 折返；ReLU(-(a*b)) 为正则惩罚
        sign_flip = F.relu(-(d1[..., 1:] * d1[..., :-1]))
        self.flip_penalty = sign_flip.mean()



        # 5) 使用 align_corners=False + 像素中心坐标的 grid_sample 进行扭曲
        #    构造 2D 网格（H=L, W=1），沿“高”维采样
        device = x.device
        x_2d = x.unsqueeze(-1)  # [B,1,L,1]

        # 像素中心坐标： (i+0.5)/L*2-1
        i = torch.arange(L, device=device, dtype=x.dtype)
        original_grid_y = ((i + 0.5) / L * 2 - 1).view(1, 1, L)  # [1,1,L]

        # 位移归一化：delta_y = total_shift * (2/L)
        shift_norm = total_shift * (2.0 / L)                     # [B,1,L]
        sp_norm = original_grid_y - shift_norm                   # [B,1,L]

        grid_x = torch.zeros((B, L, 1), device=device, dtype=x.dtype)  # x 方向恒 0
        grid_y = sp_norm.permute(0, 2, 1)                              # [B,L,1]
        grid = torch.stack([grid_x, grid_y], dim=-1)                   # [B,L,1,2]

        x_warped = F.grid_sample(
            x_2d, grid, mode='bilinear', padding_mode='border', align_corners=False
        ).squeeze(-1)  # [B,1,L]

        return x_warped, total_shift



# ========================== D) 注意力门控==========================
class AttentionGate(nn.Module):
    def __init__(self, in_channels=1, reduction=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(in_channels, max(1, in_channels // reduction)), nn.ReLU(),
            nn.Linear(max(1, in_channels // reduction), in_channels), nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=7, padding=3), nn.Sigmoid()
        )

    def forward(self, x, alpha_c=0.3, floor=0.6):
        # 兜底维度
        if x.dim() == 4 and x.size(2) == 1: x = x.squeeze(2)
        if x.dim() == 4 and x.size(-1) == 1: x = x.squeeze(-1)
        # 残差式通道注意力（围绕 1.0 小幅摆动）
        ca = self.channel_attention(x).unsqueeze(-1)           # [B,C,1]
        ca = 1.0 + alpha_c * (ca - 0.5) * 2.0
        x = x * ca
        # 带保底的空间注意力
        sa = self.spatial_attention(x)                         # [B,1,L]
        sa = floor + (1.0 - floor) * sa
        x = x * sa
        return x, sa.squeeze(1)


# ========================== C) 强度变换器 ==========================
class IntensityTransformer(nn.Module):
    def __init__(self, input_dim=1500):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([self._res(128) for _ in range(4)])
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 7, padding=3)
        )

        self.scale_head = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 1, 3, padding=1), nn.Softplus()
        )
        self.scale_base = 0.8   # 下限
        self.scale_span = 0.8   # 幅度，最终范围约 [0.8, 1.6]

    def _res(self, c):
        return nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(),
            nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c)
        )

    def forward(self, x):
        idt = x
        h = self.encoder(x)
        for blk in self.res_blocks:
            h = F.relu(blk(h) + h)
        decoded = torch.tanh(self.decoder(h))               # [-1,1] 的逐点残差
        scale_map = self.scale_base + self.scale_span * (self.scale_head(h)/(1.0+self.scale_head(h)))
        # 上式等价于“压缩后的 Softplus”，数值更稳；也可用 sigmoid 直接映射到区间
        y_raw = idt + decoded * scale_map
        y = torch.clamp(y_raw, -1.0, 2.0)   # 梯度在饱和段为 0
        return y

class SegmentGainTransformer(nn.Module):
    """
    显式段增益（OLA）版本，最小改动：
      - 输入:  x [B,1,L]
      - 输出:  y [B,1,L]，以及便于诊断的 gain_curve [B,1,L]
    """
    def __init__(self, L=1500, M=300, O=150, g_span=0.4):
        super().__init__()
        self.L = L
        self.M = M
        self.O = O
        self.step = M - O
        self.g_span = g_span  # 增益范围是 [1-g_span, 1+g_span]，默认 [0.6,1.4]


        self.seg_net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 聚成 [B*K,16,1]
            nn.Flatten(),             # [B*K,16]
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1)           # 未做激活：后面用 tanh 映射到增益范围
        )

        # Hann 窗（重叠处自然平滑）
        w = torch.hann_window(M, periodic=False)  # [M]
        self.register_buffer('hann', w)

    def forward(self, x):
        """
        x: [B,1,L]
        return:
            y: [B,1,L]
            gain_curve: [B,1,L]
        """
        assert x.dim() == 3 and x.size(1) == 1
        B, _, L = x.shape
        device = x.device
        M, step, pad = self.M, self.step, self.M // 2

        # 反射填充以对齐你现有的分段口径（和 losses 里一致）
        x_pad = F.pad(x, (pad, pad), mode='reflect')  # [B,1,L+2*pad]

        # unfold: [B,1,K,M] -> [B*K,1,M]
        segs = x_pad.unfold(-1, M, step)  # [B,1,K,M]
        K = segs.size(2)
        segs = segs.contiguous().view(B * K, 1, M)

        # 逐段估计增益标量 g_k，范围 [1-g_span, 1+g_span]
        g_raw = self.seg_net(segs).view(B, K)              # [B,K]
        g = 1.0 + self.g_span * torch.tanh(g_raw)          # 中心在 1，平滑可导

        # 用 OLA 拼出连续的增益曲线（先在 padding 域上合成，再裁回 L）
        gain_pad = torch.zeros(B, 1, L + 2 * pad, device=device)
        norm_pad = torch.zeros(1, 1, L + 2 * pad, device=device)

        w = self.hann.to(device)                           # [M]
        for t in range(K):
            s = t * step
            # 本段增益曲线 = 标量 * Hann
            g_seg = (g[:, t].view(B, 1, 1) * w.view(1, 1, M))  # [B,1,M]
            gain_pad[:, :, s:s + M] += g_seg
            norm_pad[:, :, s:s + M] += w.view(1, 1, M)

        # 防 0：重叠处归一化
        gain_pad = gain_pad / (norm_pad + 1e-8)
        gain_curve = gain_pad[:, :, pad:pad + L]           # 裁回 [B,1,L]

        # 应用段增益（纯乘法）
        y = gain_curve * x
        return y, gain_curve

# ========================== 窗口峰检测 ==========================
class WindowPeakDetector:
    def __init__(self, raman_shift, reference_spectrum, window_size=100, step=2, prominence=0.02):
        self.raman_shift = raman_shift
        self.reference_spectrum = reference_spectrum
        self.window_size = window_size
        self.step = step
        self.prominence = prominence
        self.window_mask = self._create_window_mask()

    def _create_window_mask(self):
        mask = np.zeros(len(self.raman_shift))
        start_idx = np.where(self.raman_shift >= 200)[0][0]
        end_idx = np.where(self.raman_shift <= 3200)[0][-1]
        window_count = int((self.raman_shift[end_idx] - self.raman_shift[start_idx]) // self.window_size)
        smoothed = savgol_filter(self.reference_spectrum, window_length=11, polyorder=3)

        def find_significant_peaks(x, y, min_prominence=self.prominence):
            peaks, _ = find_peaks(y, prominence=min_prominence, height=0.05, distance=5)
            return [(x[p], y[p]) for p in peaks]

        for k in range(window_count):
            s = start_idx + k * (self.window_size // self.step)
            e = min(s + (self.window_size // self.step) + 1, len(self.raman_shift))
            wn = self.raman_shift[s:e]
            ws = smoothed[s:e]
            if len(find_significant_peaks(wn, ws)) > 0:
                mask[s:e] = 1.0
        return torch.FloatTensor(mask)

    def get_mask(self):
        return self.window_mask


# ========================== ResBlock / DepthwiseSeparableConv1d ==========================
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel), nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, 3, padding=1, bias=False),
            nn.BatchNorm1d(outchannel), nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, 1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = F.relu(out + self.shortcut(x))
        return out


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))


# ========================== B) 分段增益：以 1 为中心 + Overlap-Add ==========================
class SegmentHead(nn.Module):
    """输出每段增益，约束在 [0.65, 1.35] 周围摆动"""
    def __init__(self, in_ch: int, seg_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, 5, padding=2, dilation=1), nn.ReLU(),
            nn.Conv1d(in_ch, in_ch, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(in_ch, in_ch // 2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(in_ch // 2, 1, 1)
        )
        self.seg_len = seg_len

    def forward(self, feat_seg):  # [B,in_ch,M]
        g_raw = self.net(feat_seg)               # [B,1,M]
        g = 1.0 + 0.65 * torch.tanh(g_raw)        # ∈ [0.2, 1.8] - 扩大范围  0.8-0.6减小试试
        return g


def _segment_forward(x, feat, seg_head, M=300, O=150, return_chunks=False):
    """反射填充 + Overlap-Add + Hann window，末尾裁剪回原长"""
    B, _, N = x.shape
    step = M - O
    pad = M // 2

    # 1) 反射填充
    x_pad = F.pad(x, (pad, pad), mode='reflect')
    feat_pad = F.pad(feat, (pad, pad), mode='reflect')

    # 2) 在填充后长度上做 unfold/OLA
    feat_segs = feat_pad.unfold(-1, M, step)  # [B,C,K,M]
    x_segs = x_pad.unfold(-1, M, step)        # [B,1,K,M]
    K = feat_segs.shape[2]

    y_accum = x_pad.new_zeros(B, 1, x_pad.shape[-1])
    w_accum = x_pad.new_zeros(B, 1, x_pad.shape[-1])
    win = torch.hann_window(M, device=x.device).view(1, 1, M)

    pos = 0
    segment_chunks = []
    for k in range(K):
        f_seg = feat_segs[:, :, k, :]          # [B,C,M]
        g_seg = seg_head(f_seg)                 # [B,1,M]
        y_seg = (g_seg * x_segs[:, :, k, :]) * win

        if return_chunks:
            segment_chunks.append(y_seg.squeeze(1))

        end = min(pos + M, x_pad.shape[-1])
        m = end - pos
        y_accum[:, :, pos:end] += y_seg[:, :, :m]
        w_accum[:, :, pos:end] += win[:, :, :m]
        pos += step

    y_hat_pad = y_accum / (w_accum + 1e-8)

    # 3) 裁掉两端 padding，回到原始长度 N
    y_hat = y_hat_pad[:, :, pad:pad + N]

    if return_chunks:
        segment_chunks = torch.stack(segment_chunks, dim=1)  # [B,K,M]
        return torch.clamp(y_hat, -1e6, 1e6), segment_chunks
    else:
        return torch.clamp(y_hat, -1e6, 1e6)



# ========================== 主网络 ==========================
class ImprovedRamanNet(nn.Module):
    def __init__(self, classes=2, inputlength=1500, n_residual_blocks=8, raman_shift=None, reference_spectrum=None):
        super().__init__()
        self.inputlength = inputlength
        self.data_length = inputlength

        self.xwarp = ImprovedXAxisWarp(L=inputlength, max_shift_bins=30)
        self.attention_gate = AttentionGate()
        self.multi_scale_gate = MultiScaleGate1D(inC=1, midC=24, k_list=(3, 9, 15), d_list=(1, 2, 4))
        # self.intensity_transformer = IntensityTransformer(inputlength)
        self.intensity_transformer = SegmentGainTransformer(L=1500, M=300, O=150, g_span=0.4)
        self.smoother = nn.Conv1d(1, 1, kernel_size=5, padding='same', bias=False, groups=1)

        # 初始化卷积核为均值核
        kernel = torch.full((1, 1, 5), 1.0 / 5.0)
        self.smoother.weight = nn.Parameter(kernel, requires_grad=False)


        # 窗口掩码（可训练）
        if raman_shift is not None and reference_spectrum is not None:
            peak_detector = WindowPeakDetector(raman_shift, reference_spectrum)
            init_mask = peak_detector.get_mask()
        else:
            init_mask = torch.ones(inputlength)
        self.window_mask = nn.Parameter(init_mask, requires_grad=True)


        self.premodel = nn.Sequential(
            nn.Conv1d(1, 4, 5, padding=2, bias=False), nn.ReLU(),
            nn.Conv1d(4, 8, 3, padding=1, bias=False), nn.ReLU(),
            nn.Conv1d(8, 16, 3, padding=1, bias=False), nn.ReLU()
        )

        # 下采样 + 残差
        in_ch = 16; out_ch = in_ch * 2
        model = []
        for _ in range(2):
            self.data_length = (self.data_length - 1) // 2 + 1
            model += [nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1, groups=in_ch),
                      nn.InstanceNorm1d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch; out_ch = in_ch * 2
        for _ in range(n_residual_blocks):
            model += [ResBlock(in_ch, in_ch)]
        # 上采样
        out_ch = in_ch // 2
        for _ in range(2):
            self.data_length = (self.data_length - 1) * 2 + 1
            model += [nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1, groups=out_ch),
                      nn.InstanceNorm1d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch; out_ch = in_ch // 2
        self.model = nn.Sequential(*model)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv1d(16, 32, 11, padding=5), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, classes)
        )

        self.fc1 = nn.Linear(self.inputlength, self.inputlength)
        self.pds = nn.Conv1d(in_ch, self.inputlength, 1)
        self.deepwise = DepthwiseSeparableConv1d(self.inputlength, self.inputlength)

        # 分段设置
        self.seg_len = 300
        self.seg_overlap = 150
        self.seg_head = SegmentHead(in_ch=16, seg_len=self.seg_len)
        
        # 全局强度预测器
        self.global_intensity_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):


        x1 = x.unsqueeze(1)  # 增加维度 [B, 1, L]

        # 1) 波数对齐
        x_aligned, self.last_disp = self.xwarp(x1)

        # 2) 注意力机制
        # 多尺度特征提取
        x_multiscale = self.multi_scale_gate(x_aligned)
          #  attention_gate
        x_gated, _ = self.attention_gate(x_multiscale)

        x_intensity, _ = self.intensity_transformer(x_gated)

        # 骨干特征 & 分类
        feat = self.premodel(x_intensity)
        feat = self.model(feat)                # [B,16,L]
        out_classes = self.classifier(feat)

        # 分段增益（Overlap-Add）
        trans, segment_chunks = _segment_forward(
            x_intensity, feat, self.seg_head, 
            M=self.seg_len, O=self.seg_overlap, return_chunks=True
        )
        trans = trans.squeeze(1)
        
        ones = torch.ones_like(trans)
        

        global_intensity_scale = torch.sigmoid(self.global_intensity_predictor(feat))  # [B, 1]
        trans = trans * (0.8 + 0.3 * global_intensity_scale)  # 在[0.8, 1.2]范围内调整0.8-0.4--0.85，0.25
        
        return out_classes, trans, ones, x_aligned.squeeze(1), segment_chunks


# if __name__ == '__main__':
#     data = torch.randn(4,1500)
#     model = ImprovedRamanNet(classes=3, inputlength=1500)
#     outputs = model(data)
#     class_out, trans_A2B, domain_coeff, x_gated_smoothed, segment_chunks = outputs
#     print(  class_out.shape)  
    
