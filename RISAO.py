import numpy as np
import torch
import matplotlib.pyplot as plt
import RISSatUAVCom
import FindW_WMMSE
import FindPhi_GradientAscent

class RISAlternatingOptimization:
    """RIS辅助通信系统的优化类"""
    def __init__(self, S, U, N, M, P_s, sigma2, h_su, H_sR, g_Ru, seed=42):
        """初始化系统参数和信道"""
        self.S = S  # 卫星数量
        self.U = U  # 无人机数量
        self.N = N  # 天线数量
        self.M = M  # RIS单元数量
        self.P_s = P_s  # 发射功率
        self.sigma2 = sigma2  # 噪声方差
        self.h_su = h_su  # 卫星到无人机的信道
        self.H_sR = H_sR  # 卫星到RIS的信道
        self.g_Ru = g_Ru  # RIS到无人机的信道
        self.set_seed(seed)
        self.W_su = np.zeros((self.U, self.S, self.N), dtype=complex) # 预编码矩阵
        self.theta = np.random.uniform(0, 2 * np.pi, M)  # 随机初始化RIS相位
        self.Rate = [0]  # 存储和速率

    def set_seed(self, seed):
        """设置随机种子以确保结果可重现"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def compute_equivalent_channel(self):
        """计算等效信道矩阵 H"""
        Phi = np.diag(np.exp(1j * self.theta))  # RIS相位矩阵，size: M x M
        H = np.zeros((self.S * self.N, self.U), dtype=complex)
        for u in range(self.U):
            for s in range(self.S):
                h_tilde = self.h_su[u, s, :].T + self.g_Ru[u, :] @ Phi @ self.H_sR[s, :, :].T
                H[s * self.N:(s + 1) * self.N, u] = h_tilde
        return H

    def initialize_W(self, H):
        """基于MRT初始化预编码矩阵 W"""
        W = np.zeros((self.S * self.N, self.U), dtype=complex)
        for i in range(self.U):
            h_i = H[:, i]  # 第 i 个用户的信道向量
            w_i = h_i / np.linalg.norm(h_i, 2)  # 归一化
            W[:, i] = w_i * np.sqrt(self.P_s / self.U)  # 分配功率
        return W

    def optimize_W(self, H, W_init):
        """使用WMMSE算法优化预编码矩阵 W"""
        Woptimization = FindW_WMMSE.FindW_WMMSE(self.S, self.U, self.N, self.M, H, W_init, self.P_s, self.sigma2)
        W_opt, rate_w = Woptimization.optimize()
        return W_opt, rate_w

    def optimize_theta(self, W_su, R_init):
        """使用梯度上升算法优化RIS相位 theta"""
        h_su_t = torch.tensor(self.h_su, dtype=torch.complex128).clone().detach()
        H_sR_t = torch.tensor(self.H_sR, dtype=torch.complex128).clone().detach()
        g_Ru_t = torch.tensor(self.g_Ru, dtype=torch.complex128).clone().detach()
        W_su_t = torch.tensor(W_su, dtype=torch.complex128).clone().detach()
        theta_t = torch.tensor(self.theta, dtype=torch.float64).clone().detach()
        PhiOptimization = FindPhi_GradientAscent.RISOptimization(
            self.S, self.U, self.N, self.M, h_su_t, H_sR_t, g_Ru_t, W_su_t, theta_t, R_init, self.sigma2
        )
        theta, rate_phi = PhiOptimization.optimize_theta(2000, 0.01)
        return theta, rate_phi

    def compute_Sinr_Rsum(self, W_su):
        """计算SINR和和速率"""
        exp_j_theta = np.exp(1j * self.theta)
        Phi = np.diag(exp_j_theta)  # RIS相位矩阵，size: M x M
        SINR = np.zeros(self.U)
        sigout = np.zeros(self.U, dtype=complex)
        for u in range(self.U):
            signal = 0
            for s in range(self.S):
                equiv_channel = self.h_su[u, s, :] + self.g_Ru[u, :] @ Phi @ self.H_sR[s, :, :].T
                signal += np.vdot(equiv_channel, W_su[u, s, :])
            sigout[u] = signal
            signal_power = np.abs(signal) ** 2
            interference_power = 0
            for u_prime in range(self.U):
                if u_prime != u:
                    interf = 0
                    for s in range(self.S):
                        equiv_channel = self.h_su[u, s, :] + self.g_Ru[u, :] @ Phi @ self.H_sR[s, :, :].T
                        interf += np.vdot(equiv_channel, W_su[u_prime, s, :])
                    interference_power += np.abs(interf) ** 2
            SINR[u] = signal_power / (interference_power + self.sigma2)
        R_sum = np.sum(np.log2(1 + SINR))
        return sigout, SINR, R_sum

    def run_optimization(self, max_iter=1000, tol=1e-4):
        """执行联合优化过程"""
        for iter in range(max_iter):
            H = self.compute_equivalent_channel()
            W = self.initialize_W(H) if iter == 0 else W_opt
            W_opt, rate_w = self.optimize_W(H, W)
            # self.W_su = np.zeros((self.U, self.S, self.N), dtype=complex)
            for u in range(self.U):
                for s in range(self.S):
                    self.W_su[u, s, :] = W_opt[s * self.N:(s + 1) * self.N, u]
            _, _, R_w = self.compute_Sinr_Rsum(self.W_su)
            self.Rate.append(R_w)
            theta, rate_phi = self.optimize_theta(self.W_su, 0)  # R_init=0，与原代码一致
            self.theta = theta
            sigout_phi, _, R_phi = self.compute_Sinr_Rsum(self.W_su)
            self.Rate.append(R_phi)
            print(f'迭代次数: {iter}, FindW: rate_w={rate_w[-1]}, FindPhi: rate_phi={rate_phi[-1]}')
            if iter > 0 and abs(self.Rate[-1] - self.Rate[-2]) < tol and abs(self.Rate[-2] - self.Rate[-3]) < tol:
                break
        return sigout_phi, self.Rate[-1], self.W_su, self.theta

    def plot_results(self):
        """绘制和速率曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.Rate)
        plt.ylabel('Sum Rate')
        plt.xlabel('iterations')
        plt.grid(True)
        plt.show()

def main():
    """主函数：设置参数并运行优化"""
    set_seed(42)
    S = 2  # 卫星数量
    U = 3  # 无人机数量
    N = 4  # 天线数量
    M = 16  # RIS单元数量
    P_s = 1  # 发射功率
    sigma2 = 1e-16  # 噪声方差
    gain_factor = 1e8  # 增益因子

    # 生成信道矩阵
    Sat_UAV_comm = RISSatUAVCom.RISSatUAVCom(10, U, S, N, M)
    h_su, H_sR, g_Ru = Sat_UAV_comm.setup_channel()

    # 在通信系统中，信号功率通常是∣h^H w∣^2 
    h_su = np.conj(h_su)
    H_sR = np.conj(H_sR)
    g_Ru = np.conj(g_Ru)

    # 信道矩阵放大n倍，噪声功率放大n^2倍，SINR不变，优化结果不变。
    h_su = h_su * gain_factor
    H_sR = H_sR * gain_factor
    sigma2 = sigma2 * gain_factor ** 2

    # 实例化并运行优化
    system = RISAlternatingOptimization(S, U, N, M, P_s, sigma2, h_su, H_sR, g_Ru)
    sigout, Rate, _, _ = system.run_optimization()
    system.plot_results()

def set_seed(seed):
    """全局设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    main()