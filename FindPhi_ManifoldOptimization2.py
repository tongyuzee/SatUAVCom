import numpy as np
import matplotlib.pyplot as plt

class RISOptimization:
    def __init__(self, S=2, U=3, N=4, M=8, sigma2=1e-3):
        # 参数初始化
        self.S = S  # LEO 卫星数量
        self.U = U  # UAV 数量
        self.N = N  # 天线数量
        self.M = M  # RIS 单元数量
        self.sigma2 = sigma2  # 噪声方差
        
        # 初始化信道和预编码向量
        self._initialize_channels()
        
        # 计算中间变量
        self._compute_intermediate_variables()
        
        # 优化相关变量
        self.v = np.exp(1j * np.random.uniform(0, 2 * np.pi, self.M))
        self.Rsum_history = []

    def _initialize_channels(self):
        np.random.seed(42)
        self.h_su = np.random.randn(self.U, self.S, self.N) + 1j * np.random.randn(self.U, self.S, self.N)
        # self.H_sR = np.random.randn(self.S, self.M, self.N) + 1j * np.random.randn(self.S, self.M, self.N)
        self.H_sR = np.random.randn(self.S, self.N, self.M) + 1j * np.random.randn(self.S, self.N, self.M)
        self.g_Ru = np.random.randn(self.U, self.M) + 1j * np.random.randn(self.U, self.M)
        self.w_su = np.random.randn(self.U, self.S, self.N) + 1j * np.random.randn(self.U, self.S, self.N)

    def _compute_intermediate_variables(self):
        self.a_su = np.zeros((self.S, self.U, self.M, self.N), dtype=complex)
        for s in range(self.S):
            for u in range(self.U):
                self.a_su[s, u] = np.diag(self.g_Ru[u]) @ self.H_sR[s].T

        self.b_u = np.zeros(self.U, dtype=complex)
        self.c_u = np.zeros((self.U, self.M), dtype=complex)
        self.b_uu = np.zeros((self.U, self.U), dtype=complex)
        self.c_uu = np.zeros((self.U, self.U, self.M), dtype=complex)

        for u in range(self.U):
            for s in range(self.S):
                self.b_u[u] += self.h_su[u, s].T @ self.w_su[u, s]
                self.c_u[u] += self.a_su[s, u] @ self.w_su[u, s]
                for up in range(self.U):
                    self.b_uu[u, up] += self.h_su[u, s].T @ self.w_su[up, s]
                    self.c_uu[u, up] += self.a_su[s, u] @ self.w_su[up, s]

    def compute_Rsum(self, v):
        gamma_u = np.zeros(self.U)
        for u in range(self.U):
            signal = np.abs(self.b_u[u] + v.T @ self.c_u[u])**2
            interference = 0
            for up in range(self.U):
                if up != u:
                    interference += np.abs(self.b_uu[u, up] + v.T @ self.c_uu[u, up])**2
            interference += self.sigma2
            gamma_u[u] = signal / interference
        return np.sum(np.log2(1 + gamma_u))

    def compute_gradient(self, v):
        grad = np.zeros(self.M, dtype=complex)
        for u in range(self.U):
            P_u = np.abs(self.b_u[u] + v.T @ self.c_u[u])**2
            Q_u = self.sigma2
            for up in range(self.U):
                if up != u:
                    Q_u += np.abs(self.b_uu[u, up] + v.T @ self.c_uu[u, up])**2
            gamma_u = P_u / Q_u
            
            dP_dvconj = (self.b_u[u] + v.T @ self.c_u[u]) * self.c_u[u]
            dQ_dvconj = np.zeros(self.M, dtype=complex)
            for up in range(self.U):
                if up != u:
                    dQ_dvconj += (self.b_uu[u, up] + v.T @ self.c_uu[u, up]) * self.c_uu[u, up]
            
            dgamma_dvconj = (dP_dvconj / Q_u) - (P_u / Q_u**2) * dQ_dvconj
            grad += (1 / (1 + gamma_u) / np.log(2)) * dgamma_dvconj
        return grad

    def optimize(self, max_iter=1000, step_size=0.01):
        Rsum = self.compute_Rsum(self.v)
        self.Rsum_history.append(Rsum)
        
        for iter in range(max_iter):
            if iter % 100 == 0:
                print(f"Iteration {iter}, Rsum = {Rsum}")
            
            grad_euclidean = self.compute_gradient(self.v)
            grad_manifold = grad_euclidean - np.real(np.sum(grad_euclidean * self.v)) * self.v
            self.v = self.v + step_size * grad_manifold
            self.v = self.v / np.abs(self.v)
            
            Rsum = self.compute_Rsum(self.v)
            if Rsum < self.Rsum_history[-1]:
                break
            self.Rsum_history.append(Rsum)
        
        return self.v

    def plot_results(self):
        Phi = np.diag(self.v)
        print("Optimized Phi (diagonal elements):", np.angle(np.diag(Phi)))
        print("Final Rsum:", self.compute_Rsum(self.v))
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.Rsum_history, label='Rsum')
        plt.xlabel('Iteration')
        plt.ylabel('Rsum')
        plt.title('Rsum vs. Iteration')
        plt.grid(True)
        plt.legend()
        plt.show()

# 使用类进行优化
if __name__ == "__main__":
    optimizer = RISOptimization()
    optimizer.optimize()
    optimizer.plot_results()