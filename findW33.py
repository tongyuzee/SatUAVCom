import numpy as np
import matplotlib.pyplot as plt

class WMMSEOptimizer:
    def __init__(self, S, U, N, M, P_s, sigma2, gain_factor, seed=42):
        """初始化系统参数和信道矩阵"""
        np.random.seed(seed)
        self.S = S  # 卫星数量
        self.U = U  # 无人机数量
        self.N = N  # 天线数量
        self.M = M  # RIS单元数量
        self.P_s = P_s  # 发射功率
        self.sigma2 = sigma2  # 噪声方差
        self.gain_factor = gain_factor
        
        # 生成信道矩阵
        self.h_su = np.random.randn(U, S, N) + 1j * np.random.randn(U, S, N)  # LEO到UAV信道
        self.H_sR = np.random.randn(S, N, M) + 1j * np.random.randn(S, N, M)  # LEO到RIS信道
        self.G_Ru = np.random.randn(U, M) + 1j * np.random.randn(U, M)  # RIS到UAV信道
        self.Phi = np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, M)))  # RIS相移矩阵
        
        # 计算等效信道矩阵 H
        self.H = np.zeros((S * N, U), dtype=complex)
        for u in range(U):
            for s in range(S):
                h_tilde = self.h_su[u, s, :].T + self.G_Ru[u, :] @ self.Phi @ self.H_sR[s, :, :].T
                self.H[s * N:(s + 1) * N, u] = h_tilde
        
        # 初始化预编码矩阵 W 并归一化
        self.W = np.random.randn(S * N, U) + 1j * np.random.randn(S * N, U)
        total_power = np.trace(self.W @ self.W.conj().T)
        self.W = self.W * np.sqrt(self.P_s / total_power)
    
    def compute_sum_rate(self):
        """计算和速率 R"""
        sum1 = 0
        for i in range(self.U):
            sum1 += np.vdot(self.H[:, i], self.W[:, i]) * np.vdot(self.W[:, i], self.H[:, i])
        interfere = np.zeros(self.U)
        R = 0
        for i in range(self.U):
            interfere[i] = sum1 - np.vdot(self.H[:, i], self.W[:, i]) * np.vdot(self.W[:, i], self.H[:, i])
            INR = self.sigma2 + interfere[i]
            sinal = np.vdot(self.H[:, i], self.W[:, i]) * np.vdot(self.W[:, i], self.H[:, i])
            R += np.log2(1 + (sinal / INR))
        return R
    
    def generate_G(self):
        """生成接收滤波器 G"""
        sum1 = 0
        for i in range(self.U):
            sum1 += np.vdot(self.H[:, i], self.W[:, i]) * np.vdot(self.W[:, i], self.H[:, i])
        G = np.zeros(self.U, dtype=complex)
        for i in range(self.U):
            G[i] = np.vdot(self.H[:, i], self.W[:, i]) / (sum1 + self.sigma2)
        return G
    
    def generate_La(self):
        """生成权重 La"""
        sum1 = 0
        for i in range(self.U):
            sum1 += np.vdot(self.H[:, i], self.W[:, i]) * np.vdot(self.W[:, i], self.H[:, i])
        La = np.zeros(self.U)
        for i in range(self.U):
            temp = np.vdot(self.W[:, i], self.H[:, i]) * (1 / (sum1 + self.sigma2)) * np.vdot(self.H[:, i], self.W[:, i])
            La[i] = 1 / (1 - temp)
        return La
    
    def generate_W(self, G, La):
        """生成预编码向量 W"""
        S_N = self.S * self.N
        sum2 = np.zeros((S_N, S_N), dtype=complex)
        for i in range(self.U):
            sum2 += np.outer(self.H[:, i], self.H[:, i].conj()) * G[i] * La[i] * np.conj(G[i])
        
        mu_max = 10
        mu_min = 0
        iter_max = 100
        for iter in range(iter_max):
            mu = (mu_min + mu_max) / 2
            Pt = 0
            W_opt = np.zeros((S_N, self.U), dtype=complex)
            for i in range(self.U):
                A = sum2 + mu * np.eye(S_N)
                W_opt[:, i] = np.linalg.solve(A, self.H[:, i]) * G[i] * La[i]
                Pt += np.real(np.trace(np.outer(W_opt[:, i], W_opt[:, i].conj())))
            if Pt > self.P_s:
                mu_min = mu
            else:
                mu_max = mu
            if abs(mu_max - mu_min) < 1e-5:
                break
        print(f'求解最优mu共迭代{iter}次, mu*={mu}, P={Pt}')
        return W_opt
    
    def optimize(self, max_iter=200, tol=1e-5):
        """执行WMMSE算法的迭代优化"""
        rate = []
        for iter in range(max_iter):
            R_pre = self.compute_sum_rate()
            rate.append(R_pre)
            G = self.generate_G()
            La = self.generate_La()
            self.W = self.generate_W(G, La)
            R = self.compute_sum_rate()
            if abs(R - R_pre) < tol:
                print(f'求解和速率共迭代{iter}次')
                break
        rate.append(R)
        return rate
    
    def plot_rate(self, rate):
        """绘制和速率随迭代次数的变化"""
        plt.plot(rate)
        plt.ylabel('Sum Rate')
        plt.xlabel('iterations')
        plt.show()

# 使用示例
S = 2  # 卫星数量
U = 3  # 无人机数量
N = 4  # 天线数量
M = 5  # RIS单元数量
P_s = 1  # 发射功率
sigma2 = 1e-4  # 噪声方差
gain_factor = 20.0  # 增益因子

optimizer = WMMSEOptimizer(S, U, N, M, P_s, sigma2, gain_factor)
rate = optimizer.optimize()
optimizer.plot_rate(rate)