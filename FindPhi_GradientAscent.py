import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RISOptimization:
    def __init__(self, S, U, N, M, h_su, H_sR, g_Ru, W_su, theta_init, R_init=0, sigma2=1e-3):
        # 系统参数
        self.S = S  # LEO 卫星数量
        self.U = U  # UAV 数量
        self.N = N  # 每颗 LEO 卫星的天线数量
        self.M = M  # RIS 反射单元数量
        self.sigma2 = torch.tensor(sigma2, dtype=torch.float64)  # 噪声功率

        # 初始化信道矩阵和变量
        self.h_su = h_su  # h_{u,s}: U x S x N (相当于转置后的 1 x N)
        self.H_sR = H_sR  # H_{s,R}: S x N x M (转置后为 M x N)
        self.g_Ru = g_Ru  # g_{R,u}: U x M (相当于转置后的 1 x M)
        self.w_su = W_su  # w_{u,s}: U x S x N
        self.theta_init = theta_init  # 初始的 theta
        self.R_init = R_init  # 初始的 R

    def compute_Rsum(self, theta):
        # 构造 Phi 矩阵：Phi = diag(e^{j*theta})
        exp_j_theta = torch.exp(1j * theta)
        Phi = torch.diag(exp_j_theta)  # (M, M) 的对角矩阵

        SINR = torch.zeros(self.U, dtype=torch.float64)
        for u in range(self.U):
            # 信号部分 (分子)
            signal = torch.tensor(0.0, dtype=torch.complex128)
            for s in range(self.S):
                # 等效信道: h_{s,u}^T + G_{R,u}^T Phi H_{s,R}^T
                # 注意 h_su 的索引变为 [u, s, :] 因为形状改为 (U, S, N)
                equiv_channel = self.h_su[u, s, :]+ self.g_Ru[u, :]@ Phi @ self.H_sR[s, :, :].T
                signal += torch.vdot(equiv_channel , self.w_su[u, s, :])
            signal_power = torch.abs(signal) ** 2

            # 干扰部分 (分母)
            interference_power = torch.tensor(0.0, dtype=torch.complex128)
            for u_prime in range(self.U):
                if u_prime != u:
                    interference_sum = torch.tensor(0.0, dtype=torch.complex128)
                    for s in range(self.S):
                        equiv_channel = self.h_su[u, s, :] + self.g_Ru[u, :] @ Phi @ self.H_sR[s, :, :].T
                        interference_sum += torch.vdot(equiv_channel, self.w_su[u_prime, s, :])
                    interference_power += torch.abs(interference_sum) ** 2
            # SINR
            SINR[u] = torch.abs(signal_power / (interference_power + self.sigma2))
            # 计算 R_sum
        R_sum = torch.sum(torch.log2(1 + SINR))
        return R_sum

    def optimize_theta(self, max_iter=2000, learning_rate=0.01):
        # 初始化 theta 为可优化的参数
        theta = self.theta_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=learning_rate)
        R_sum_history = [self.R_init]
        for iter in range(max_iter):
            optimizer.zero_grad()
            R_sum = self.compute_Rsum(theta)
            (-R_sum).backward()  # 取负值以最大化 R_sum
            optimizer.step()

            # 确保 theta 在 [0, 2pi) 范围内
            with torch.no_grad():
                theta.copy_(theta % (2 * torch.pi))

            R_sum_history.append(R_sum.item())

            if R_sum_history[-1] - R_sum_history[-2] < -1e4:
                raise ValueError("FindPhi: Reward is decreasing!")

            if np.abs(R_sum_history[-1] - R_sum_history[-2]) < 1e-5:
                break

            # if (iteration + 1) % 20 == 0:
            #     print(f"Iteration {iteration + 1}/{max_iter}, R_sum = {R_sum.item():.4f}")
        # print(f'FindPhi: iter={iter:03d}, R_phi = {R_sum.item():.5f}')

        return theta.detach().numpy(), R_sum_history

    def run(self, max_iter=2000, learning_rate=0.01):
        # 运行优化
        theta_opt, R_sum_history = self.optimize_theta(max_iter, learning_rate)

        # 计算最终的 R_sum
        final_Rsum = self.compute_Rsum(torch.tensor(theta_opt, dtype=torch.float64))
        print(f"\nOptimized R_sum: {final_Rsum.item():.4f}")
        print("Optimized theta (in radians):")
        print(theta_opt)

    def plot_results(self, R_sum_history):
        # 绘制结果
        plt.figure(figsize=(8, 6))
        plt.plot(R_sum_history[1:], label="R_sum")
        plt.xlabel("Iteration")
        plt.ylabel("Sum Rate (R_sum)")
        plt.title("Sum Rate vs. Iteration")
        plt.grid(True)
        plt.legend()
        plt.show()

# 创建实例并运行
if __name__ == "__main__":

    S = 2  # 卫星数量
    U = 3  # 无人机数量
    N = 4  # 天线数量
    M = 16  # RIS单元数量
    P_s = 1  # 发射功率
    sigma2 = 1e-3  # 噪声方差

    set_seed(42)

    # 初始化信道矩阵和变量
    h_su = torch.randn(U, S, N, dtype=torch.complex128)  # 修改为 (U, S, N)
    g_Ru = torch.randn(U, M, dtype=torch.complex128)
    H_sR = torch.randn(S, N, M, dtype=torch.complex128)
    W_su = torch.randn(U, S, N, dtype=torch.complex128)

    theta_init = torch.tensor(np.random.uniform(0, 2 * np.pi, M), dtype=torch.float64, requires_grad=True)

    optimizer = RISOptimization(S, U, N, M, h_su, H_sR, g_Ru, W_su, theta_init, R_init=0, sigma2=1e-3)
    theta_opt, R_sum_history = optimizer.optimize_theta(max_iter=2000, learning_rate=0.01)
    optimizer.plot_results(R_sum_history)