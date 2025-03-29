import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FindPhi_GA:
    def __init__(self, S, U, N, M, h_su, H_sR, g_Ru, W_su, theta_init, R_init=0, sigma2=1e-3):
        # 将所有张量移动到 GPU
        self.device = device
        self.S = S  # LEO 卫星数量
        self.U = U  # UAV 数量
        self.N = N  # 每颗 LEO 卫星的天线数量
        self.M = M  # RIS 反射单元数量
        self.sigma2 = torch.tensor(sigma2, dtype=torch.float64, device=self.device)  # 噪声功率

        # 初始化信道矩阵和变量并移动到 GPU
        self.h_su = h_su.to(self.device)  # h_{u,s}: U x S x N
        self.H_sR = H_sR.to(self.device)  # H_{s,R}: S x N x M
        self.g_Ru = g_Ru.to(self.device)  # g_{R,u}: U x M
        self.w_su = W_su.to(self.device)  # w_{u,s}: U x S x N
        self.theta_init = theta_init.to(self.device)  # 初始的 theta
        self.R_init = R_init  # 初始的 R

    def compute_Rsum(self, theta):
        theta = theta.to(self.device)
        # 构造 Phi 矩阵：Phi = diag(e^{j*theta})
        exp_j_theta = torch.exp(1j * theta)
        Phi = torch.diag(exp_j_theta)  # (M, M) 的对角矩阵

        SINR = torch.zeros(self.U, dtype=torch.float64, device=self.device)
        for u in range(self.U):
            # 信号部分 (分子)
            signal = torch.tensor(0.0, dtype=torch.complex128, device=self.device)
            for s in range(self.S):
                equiv_channel = self.h_su[u, s, :] + self.g_Ru[u, :] @ Phi @ self.H_sR[s, :, :].T
                signal += torch.vdot(equiv_channel, self.w_su[u, s, :])
            signal_power = torch.abs(signal) ** 2

            # 干扰部分 (分母)
            interference_power = torch.tensor(0.0, dtype=torch.float64, device=self.device)
            for u_prime in range(self.U):
                if u_prime != u:
                    interference_sum = torch.tensor(0.0, dtype=torch.complex128, device=self.device)
                    for s in range(self.S):
                        equiv_channel = self.h_su[u, s, :] + self.g_Ru[u, :] @ Phi @ self.H_sR[s, :, :].T
                        interference_sum += torch.vdot(equiv_channel, self.w_su[u_prime, s, :])
                    interference_power += torch.abs(interference_sum) ** 2
            
            # 添加数值稳定性检查
            denom = interference_power + self.sigma2
            if denom <= 1e-10:
                denom = torch.tensor(1e-10, dtype=torch.float64, device=self.device)
            # SINR
            SINR[u] = torch.abs(signal_power / denom)
        R_sum = torch.sum(torch.log2(1 + SINR))
        return R_sum

    def optimize_theta(self, max_iter=2000, learning_rate=0.01):
        theta = self.theta_init.clone().detach().requires_grad_(True).to(self.device)
        optimizer = torch.optim.Adam([theta], lr=learning_rate)
        best_R = -float('inf')
        best_theta = theta.clone()
        R_sum_history = []
        for iter in range(max_iter):
            optimizer.zero_grad()
            R_sum = self.compute_Rsum(theta)
            if R_sum > best_R:
                best_R = R_sum
                best_theta = theta.clone()
            if not torch.isfinite(R_sum):
                print(f"Warning: Non-finite R_sum detected at iteration {iter}")
                break

            (-R_sum).backward()
            # # 梯度剪裁，防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_([theta], max_norm=1.0)
            if not torch.isfinite(theta.grad).all():
                print(f"Iter {iter}: Invalid gradients, using previous best theta")
                theta.copy_(best_theta)
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            # 确保 theta 在 [0, 2pi) 范围内
            with torch.no_grad():
                theta.copy_(theta % (2 * torch.pi))

            R_sum_history.append(R_sum.item())

            if len(R_sum_history) > 1 and not np.isfinite(R_sum_history[-1]):
                print(f"Warning: Non-finite R_sum history detected at iteration {iter}")
                break

            if len(R_sum_history) > 5:
                if R_sum_history[-1] - R_sum_history[-2] < -1e4:
                    raise ValueError("FindPhi: Reward is decreasing!")
                recent_errors = [abs(R_sum_history[i] - R_sum_history[i - 1]) for i in range(-5, 0)]
                if all(error < 1e-3 for error in recent_errors):
                    break

            # if (iteration + 1) % 20 == 0:
            #     print(f"Iteration {iteration + 1}/{max_iter}, R_sum = {R_sum.item():.4f}")
        # print(f'FindPhi: iter={iter:03d}, R_phi = {R_sum.item():.5f}')
        return theta.detach().cpu().numpy(), R_sum_history

    def run(self, max_iter=2000, learning_rate=0.01):
        theta_opt, R_sum_history = self.optimize_theta(max_iter, learning_rate)
        final_Rsum = self.compute_Rsum(torch.tensor(theta_opt, dtype=torch.float64, device=self.device))
        print(f"\nOptimized R_sum: {final_Rsum.item():.4f}")
        print("Optimized theta (in radians):")
        print(theta_opt)

    def plot_results(self, R_sum_history):
        """绘制和速率曲线和误差曲线在同一张图上，使用双纵轴"""
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # 计算误差（相邻迭代之间的和速率变化）
        errors = []
        for i in range(1, len(R_sum_history)):
            errors.append(abs(R_sum_history[i] - R_sum_history[i-1]))
        
        # 确保误差不为零，避免对数坐标出现问题
        for i in range(len(errors)):
            if errors[i] < 1e-10:  # 设置一个最小值
                errors[i] = 1e-10
        
        # 和速率曲线 (左纵轴)
        color = 'tab:blue'
        ax1.set_xlabel('Iterations', fontsize=14)
        ax1.set_ylabel('Sum Rate (bps/Hz)', color=color, fontsize=14)
        line1 = ax1.plot(range(len(R_sum_history)), R_sum_history, color=color, 
                         marker='o', markersize=4, linestyle='-', linewidth=2, 
                         label='Sum Rate')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 创建右侧纵轴用于误差曲线，使用对数坐标
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Error (absolute difference)', color=color, fontsize=14)
        ax2.set_yscale('log')  # 设置为对数坐标
        line2 = ax2.plot(range(1, len(R_sum_history)), errors, color=color, 
                         marker='s', markersize=4, linestyle='--', linewidth=2,
                         label='Error')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 添加图例 - 合并两个轴的图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower center', fontsize=12)
        
        # 添加网格线
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if not os.path.exists('fig'):
            os.makedirs('fig')
        plt.savefig('fig/FindPhi_GA.pdf', format='pdf', bbox_inches='tight')
        plt.savefig('fig/FindPhi_GA.svg', format='svg', bbox_inches='tight')
        plt.savefig('fig/FindPhi_GA_with_errors.png', dpi=300, bbox_inches='tight')
        
        plt.show()

# 创建实例并运行
if __name__ == "__main__":
    S = 2  # 卫星数量
    U = 3  # 无人机数量
    N = 4  # 天线数量
    M = 16  # RIS单元数量
    P_s = 1  # 发射功率
    sigma2 = 1e-4  # 噪声方差

    set_seed(42)

    # 初始化张量并移动到 GPU
    h_su = torch.randn(U, S, N, dtype=torch.complex128, device=device)
    g_Ru = torch.randn(U, M, dtype=torch.complex128, device=device)
    H_sR = torch.randn(S, N, M, dtype=torch.complex128, device=device)
    W_su = torch.randn(U, S, N, dtype=torch.complex128, device=device)
    W_su = W_su / torch.norm(W_su) * torch.sqrt(torch.tensor(P_s, dtype=torch.float64, device=device))
    theta_init = torch.tensor(np.random.uniform(0, 2 * np.pi, M), dtype=torch.float64, device=device, requires_grad=True)

    optimizer = FindPhi_GA(S, U, N, M, h_su, H_sR, g_Ru, W_su, theta_init, R_init=0, sigma2=1e-3)
    theta_opt, R_sum_history = optimizer.optimize_theta(max_iter=2000, learning_rate=0.01)
    optimizer.plot_results(R_sum_history)