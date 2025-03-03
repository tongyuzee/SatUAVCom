import numpy as np
import cvxpy as cp

# 参数设置
S = 2       # 卫星数
U = 3       # 无人机数
M = 16      # RIS反射单元数
N = 4       # 卫星天线数
P_s = 10    # 每个卫星的功率限制 (W)
sigma2 = 0.1  # 噪声功率 (W)
max_iter = 50  # 最大迭代次数
epsilon = 1e-3  # 收敛阈值

# 生成信道矩阵 (示例为随机复数信道)
np.random.seed(0)
h = {s: {u: np.random.randn(N, 1) + 1j*np.random.randn(N, 1) for u in range(U)} for s in range(S)}
H_R = {s: np.random.randn(N, M) + 1j*np.random.randn(N, M) for s in range(S)}
G_R = {u: np.random.randn(M, 1) + 1j*np.random.randn(M, 1) for u in range(U)}
Phi = np.diag(np.exp(1j * np.random.uniform(0, 2*np.pi, M)))  # RIS相位矩阵

# 初始化波束赋形向量
w = {s: {u: np.random.randn(N, 1) + 1j*np.random.randn(N, 1) for u in range(U)} for s in range(S)}
for s in range(S):
    norm_factor = np.sqrt(P_s / sum(np.linalg.norm(w[s][u])**2 for u in range(U)))
    for u in range(U):
        w[s][u] = w[s][u] * norm_factor

# 迭代优化
R_sum_prev = 0
for iter in range(max_iter):
    # 计算等效信道 C_{s,u} = h_{s,u}^T + G_R^H @ Phi @ H_R^T  size:C^1XN
    C = {s: {u: h[s][u].T + G_R[u].conj().T @ Phi @ H_R[s].T for u in range(U)} for s in range(S)}
    
    # 计算SINR和均方误差
    gamma = np.zeros(U)
    e = np.zeros(U)
    alpha = np.zeros(U)
    beta = np.zeros(U)
    for u in range(U):
        signal = sum(C[s][u] @ w[s][u] for s in range(S)).item()
        interference = sum(np.abs(sum(C[s][u] @ w[s][u_prime] for s in range(S)))**2 for u_prime in range(U) if u_prime != u).item()
        gamma[u] = (np.abs(signal)**2) / (interference + sigma2)
        e[u] = 1 / (1 + gamma[u])
        alpha[u] = 1 + gamma[u]
        beta[u] = (np.sqrt(alpha[u]) * signal) / (interference + sigma2)

    
    # # 更新辅助变量 alpha 和 beta
    # alpha = 1 / e_u  # 简化示例，实际需按公式计算
    # beta = (np.sqrt(alpha) * signal) / (sum(np.abs(sum(C[s][u].conj().T @ w[s][u] for s in range(S)))**2 for s in range(S)) + sigma2)
    
    # 构建优化问题 (使用CVXPY)
    w_opt = {s: {u: cp.Variable((N, 1), complex=True) for u in range(U)} for s in range(S)}
    objective = 0
    constraints = []
    for s in range(S):
        for u in range(U):
            C_su = C[s][u]
            objective += alpha[u] * cp.abs(beta[u])**2 * cp.sum_squares(C_su @ w_opt[s][u]) - 2 * cp.real(beta[u].conj() * C_su @ w_opt[s][u])
        # 功率约束
        constraints.append(cp.sum_squares(cp.hstack([w_opt[s][u] for u in range(U)])) <= P_s)
    
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.SCS)
    
    # 更新波束赋形向量
    for s in range(S):
        for u in range(U):
            w[s][u] = w_opt[s][u].value
    
    # 计算总速率
    R_sum = sum(np.log2(1 + gamma[u]) for u in range(U))
    print(f"Iter {iter+1}: R_sum = {R_sum:.3f} bps/Hz")
    
    # 收敛判断
    if abs(R_sum - R_sum_prev) < epsilon:
        break
    R_sum_prev = R_sum

print("优化完成。")