import numpy as np
import matplotlib.pyplot as plt

class FindW_WMMSE:
    def __init__(self, S, U, N, M, H, W_init, P_s, sigma2):
        """初始化系统参数和信道矩阵"""
        self.S = S  # 卫星数量
        self.U = U  # 无人机数量
        self.N = N  # 天线数量
        self.M = M  # RIS单元数量
        self.P_s = P_s  # 发射功率
        self.sigma2 = sigma2  # 噪声方差
                
        # 计算等效信道矩阵 H ， size: S*N x U
        self.H = H
        
        # 初始化预编码矩阵 W 并归一化， size: S*N x U
        self.W = W_init
    
    def compute_sum_rate(self):
        """计算和速率 R"""
        R = 0
        for u in range(self.U):
            signal = np.vdot(self.H[:, u], self.W[:, u]) * np.vdot(self.W[:, u], self.H[:, u])
            interfere = 0
            for u_prime in range(self.U):
                if u_prime != u:
                    interfere += np.vdot(self.H[:, u], self.W[:, u_prime]) * np.vdot(self.W[:, u_prime], self.H[:, u])
            INR = self.sigma2 + interfere
            R += np.log2(1 + (signal / INR))
        return np.abs(R)
    
    def generate_G(self):
        """生成接收滤波器 G"""
        G = np.zeros(self.U, dtype=complex)
        for u in range(self.U):
            # 计算信号功率 |<H_u, W_u>|^2
            signal = np.vdot(self.H[:, u], self.W[:, u]) * np.vdot(self.W[:, u], self.H[:, u])
            # 计算干扰功率 \sum_{u'=1, u'\=u}^U |<H_u, W_u'>|^2
            interfere = 0
            for u_prime in range(self.U):
                if u_prime != u:
                    interfere += np.vdot(self.H[:, u], self.W[:, u_prime]) * np.vdot(self.W[:, u_prime], self.H[:, u])
            # 
            G[u] = np.vdot(self.H[:, u], self.W[:, u]) / (signal + interfere + self.sigma2)
        return G
    
    def generate_La(self):
        """生成均方误差权重 La"""
        La = np.zeros(self.U)
        for u in range(self.U):
            # 计算信号功率 |<H_u, W_u>|^2
            signal = np.vdot(self.H[:, u], self.W[:, u]) * np.vdot(self.W[:, u], self.H[:, u])
            # 计算总功率 \sum_{u'=1}^U |<H_u, W_u'>|^2 + sigma^2
            total_power = 0
            for u_prime in range(self.U):
                total_power += np.vdot(self.H[:, u], self.W[:, u_prime]) * np.vdot(self.W[:, u_prime], self.H[:, u])
            total_power += self.sigma2
            # 计算 MSE_i = 1 - signal / total_power
            MSE_i = 1 - signal / total_power
            # 计算 La[i] = 1 / MSE_i
            La[u] = 1 / (1 - np.abs(MSE_i))
        return La
    
    def generate_W(self, G, La):
        """生成预编码向量 W"""
        S_N = self.S * self.N
        sum2 = np.zeros((S_N, S_N), dtype=complex)
        for u_prime in range(self.U):
            sum2 += np.outer(self.H[:, u_prime], self.H[:, u_prime].conj()) * G[u_prime] * La[u_prime] * np.conj(G[u_prime])
        
        mu_max = 10
        mu_min = 0
        iter_max = 100
        for iter in range(iter_max):
            mu = (mu_min + mu_max) / 2
            P_current = 0
            W_opt = np.zeros((S_N, self.U), dtype=complex)
            for u in range(self.U):
                A = sum2 + mu * np.eye(S_N)
                W_opt[:, u] = np.linalg.solve(A, self.H[:, u] * G[u] * La[u])
                P_current += np.real(np.trace(np.outer(W_opt[:, u], W_opt[:, u].conj())))
            # print(f'iter={iter}, mu*={mu}, P={P_current}')
            if P_current > self.P_s:
                mu_min = mu
            else:
                mu_max = mu
            if abs(mu_max - mu_min) < 1e-5:
                break
        # print(f'求解最优mu共迭代{iter}次, mu*={mu}, P={P_current}')
        return W_opt
    
    def optimize(self, max_iter=200, tol=1e-4):
        """执行WMMSE算法的迭代优化"""
        rate = []
        R_pre = self.compute_sum_rate()
        rate.append(R_pre)
        for iter in range(max_iter):
            # R_pre = self.compute_sum_rate()
            # rate.append(R_pre)
            G = self.generate_G()
            La = self.generate_La()
            self.W = self.generate_W(G, La)
            R = self.compute_sum_rate()
            rate.append(R)
            if abs(R - rate[-2]) < tol:
                break
        print(f'求解和速率共迭代{iter}次, R={R}')
        return self.W, rate
    
    def plot_rate(self, rate):
        """绘制和速率随迭代次数的变化"""
        plt.plot(rate)
        plt.ylabel('Sum Rate')
        plt.xlabel('iterations')
        plt.show()

if __name__ == '__main__':
    S = 2  # 卫星数量
    U = 3  # 无人机数量
    N = 4  # 天线数量
    M = 16  # RIS单元数量
    P_s = 1  # 发射功率
    sigma2 = 1e-4  # 噪声方差
    # gain_factor = 20.0  # 增益因子

    np.random.seed(42)

    # 生成信道矩阵
    h_su = np.random.randn(U, S, N) + 1j * np.random.randn(U, S, N)
    H_sR = np.random.randn(S, N, M) + 1j * np.random.randn(S, N, M)
    g_Ru = np.random.randn(U, M) + 1j * np.random.randn(U, M)

    # 初始化RIS矩阵 Phi, size: M x M
    Phi = np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, M)))  

    # 计算等效信道矩阵 H ， size: S*N x U
    H = np.zeros((S * N, U), dtype=complex)
    for u in range(U):
        for s in range(S):
            h_tilde = h_su[u, s, :].T + g_Ru[u, :] @ Phi @ H_sR[s, :, :].T
            H[s * N:(s + 1) * N, u] = h_tilde

    # 信道矩阵放大n倍，噪声功率放大n^2倍，SINR不变，优化结果不变。
    H = H * 2
    sigma2 = sigma2 * 4

    # 初始化预编码矩阵 W 并归一化， size: S*N x U
    W_init = H / np.linalg.norm(H, 'fro') * np.sqrt(P_s)

    optimizer = FindW_WMMSE(S, U, N, M, H, W_init, P_s, sigma2)
    W_opt, rate = optimizer.optimize()
    W_su = np.zeros((U, S, N), dtype=complex)
    for u in range(U):
        for s in range(S):
            W_su[u, s, :] = W_opt[s * N:(s + 1) * N, u]
    print(W_opt, rate)
    optimizer.plot_rate(rate)