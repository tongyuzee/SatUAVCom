import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True

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
            La[u] = 1 / (np.abs(MSE_i))
        return La
    
    def generate_W(self, G, La):
        """生成预编码向量 W，满足所有卫星对所有用户的全局总功率约束"""
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
    
    def generate_W3(self, G, La):
        """生成预编码向量 W，满足每颗卫星的单独功率约束"""
        S_N = self.S * self.N
        N = self.N
        sum2 = np.zeros((S_N, S_N), dtype=complex)
        for u_prime in range(self.U):
            sum2 += np.outer(self.H[:, u_prime], self.H[:, u_prime].conj()) * G[u_prime] * La[u_prime] * np.conj(G[u_prime])
        
        # 假设每颗卫星的功率约束为 P_s / S（可根据需要调整为不同的 P_s^{(s)}）
        P_s_array = np.ones(self.S) * self.P_s / self.S
        # 初始化每个卫星的拉格朗日乘子 mu_s
        mu_s = np.zeros(self.S)
        mu_min = np.zeros(self.S)
        mu_max = np.ones(self.S) * 10  # 初始上界设为10
        
        iter_max = 200
        W_opt = np.zeros((S_N, self.U), dtype=complex)
        
        for iter in range(iter_max):
            # 构建正则化矩阵 A
            A = sum2.copy()
            for s in range(self.S):
                A[s * N:(s + 1) * N, s * N:(s + 1) * N] += mu_s[s] * np.eye(N)
            
            # 求解 W_opt
            for u in range(self.U):
                W_opt[:, u] = np.linalg.solve(A, self.H[:, u] * G[u] * La[u])
            
            # 计算每颗卫星的功率
            P_current = np.zeros(self.S)
            for s in range(self.S):
                for u in range(self.U):
                    W_s_u = W_opt[s * N:(s + 1) * N, u]
                    P_current[s] += np.real(np.vdot(W_s_u, W_s_u))
            
            # 更新 mu_s 使用二分搜索
            converged = True
            for s in range(self.S):
                if  abs(P_current[s] - P_s_array[s]) > 1e-3 and mu_max[s] - mu_min[s] > 1e-5:
                    converged = False
                    if P_current[s] > P_s_array[s]:
                        mu_min[s] = mu_s[s]
                        mu_s[s] = (mu_s[s] + mu_max[s]) / 2
                    else:
                        mu_max[s] = mu_s[s]
                        mu_s[s] = (mu_min[s] + mu_s[s]) / 2
            # print(f"迭代次数: {iter}, 每个卫星的功率分布:")
            # for s in range(S):
            #     print(f"卫星 {s+1}: mu={mu_s[s]:.10f} 功率：{P_current[s]:.10f}")
            if converged:
                break
         # 最终调整
        # for s in range(S):
        #     if P_current[s] > P_s_array[s]:
        #         scale = np.sqrt(P_s_array[s] / P_current[s])
        #         W_opt[s*N:(s+1)*N, :] *= scale   
        # print(f"迭代次数: {iter}, 每颗卫星功率: {P_current}")
        return W_opt

    def generate_W2(self, G, La):
        """使用 CVXPY 生成预编码向量 W"""
        S_N = self.S * self.N  # 矩阵维度：卫星天线总数
        U = self.U            # 用户（无人机）数量
        H = self.H            # 信道矩阵
        P_s = self.P_s        # 总功率约束

        # 计算 A 矩阵：sum_{u} La[u] * |G[u]|^2 * H[:, u] @ H[:, u].conj().T
        A = np.zeros((S_N, S_N), dtype=complex)
        for u in range(U):
            A += La[u] * np.abs(G[u])**2 * np.outer(H[:, u], H[:, u].conj())

        # 计算 B 矩阵：B[:, u] = La[u] * conj(G[u]) * H[:, u]
        B = np.zeros((S_N, U), dtype=complex)
        for u in range(U):
            B[:, u] = La[u] * np.conj(G[u]) * H[:, u]

        # 定义 CVXPY 优化变量 W，复数矩阵
        W = cp.Variable((S_N, U), complex=True)

        # 定义目标函数
        objective = cp.Minimize(cp.real(cp.trace(cp.conj(W).T @ A @ W)) - 2 * cp.real(cp.trace(cp.conj(B).T @ W)))

        # 定义功率约束
        constraints = [cp.sum([cp.norm(W[:, u], 2)**2 for u in range(U)]) <= P_s]

        # 构造并求解优化问题
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # 返回优化后的预编码矩阵 W
        return W.value
    
    def generate_W4(self, G, La):
        """生成预编码向量 W，支持每颗卫星单独功率约束"""
        S_N = self.S * self.N
        W_opt = cp.Variable((S_N, self.U), complex=True)
        
        # 目标函数：加权MSE最小化
        objective = 0
        for u in range(self.U):
            for u_prime in range(self.U):
                objective += La[u] * cp.square(cp.abs(G[u] * (cp.conj(self.H[:, u]).T @ W_opt[:, u_prime]) - (1 if u == u_prime else 0)))
        
        # 约束：每颗卫星的功率 <= P_s / S（均匀分配）
        constraints = []
        for s in range(self.S):
            # 提取卫星s的预编码部分（W_opt[s*N:(s+1)*N, :]）
            W_s = W_opt[s*self.N : (s+1)*self.N, :]
            constraints.append(cp.sum_squares(W_s) <= self.P_s / self.S)
        
        # 求解问题
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.SCS, verbose=False)
        
        if W_opt.value is None:
            raise ValueError("CVXPY failed to find a solution.")
        
        return W_opt.value
    
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
            self.W = self.generate_W4(G, La)
            R = self.compute_sum_rate()
            rate.append(R)
            if rate[-1] - rate[-2] < -1e-4:
                # raise ValueError("FindW: Reward is decreasing!")
                print(f"FindW: Reward is decreasing at iteration {iter}")
                break
            if abs(rate[-1] - rate[-2]) < tol:
                break
        # print(f'FindW: iter={iter:03d}, R_w = {R:.5f}')
        return self.W, rate
    
    def plot_rate(self, rate):
        """绘制和速率随迭代次数的变化"""
        plt.figure(figsize=(8, 6))
        plt.plot(rate)
        plt.ylabel('Sum Rate', fontsize=14)
        plt.xlabel('Iterations', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title('Sum Rate vs. Iteration', fontsize=14)
        plt.grid(True)
        # 创建保存目录
        if not os.path.exists('fig'):
            os.makedirs('fig')
        plt.savefig('fig/FindW_WMMSE.pdf', format='pdf', bbox_inches='tight')
        plt.savefig('fig/FindW_WMMSE.svg', format='svg', bbox_inches='tight')
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
    # 将全局预编码矩阵重塑为每个用户、每个卫星的预编码矩阵
    W_su = np.zeros((U, S, N), dtype=complex)
    for u in range(U):
        for s in range(S):
            W_su[u, s, :] = W_opt[s * N:(s + 1) * N, u]
    
    # 计算每个卫星的功率
    satellite_power = np.zeros(S)
    for s in range(S):
        for u in range(U):
            power = np.vdot(W_su[u, s, :], W_su[u, s, :]).real
            satellite_power[s] += power
    
    # 计算每个用户的功率
    user_power = np.zeros(U)
    for u in range(U):
        for s in range(S):
            power = np.vdot(W_su[u, s, :], W_su[u, s, :]).real
            user_power[u] += power
    
    # 计算总功率
    total_power = np.sum(satellite_power)
    
    print("每个卫星的功率分布:")
    for s in range(S):
        print(f"卫星 {s+1}: {satellite_power[s]:.6f} ({satellite_power[s]/total_power*100:.2f}%)")
    
    print("\n每个用户的功率分布:")
    for u in range(U):
        print(f"用户 {u+1}: {user_power[u]:.6f} ({user_power[u]/total_power*100:.2f}%)")
    
    print(f"\n总功率: {total_power:.6f}")
    print(f"期望总功率: {P_s}")
    
    # 显示总和速率
    print(f"最终和速率: {rate[-1]:.6f}")
    
    # 绘制和速率随迭代次数的变化
    optimizer.plot_rate(rate)