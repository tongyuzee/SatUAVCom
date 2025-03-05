import numpy as np

np.random.seed(42)

# 系统参数
S = 2  # 卫星数量
U = 3  # 无人机数量
N = 4  # 天线数量
M = 5  # RIS单元数量
P_s = 10.0
sigma2 = 1e-4
gain_factor = 20.0

# 随机初始化信道矩阵
np.random.seed(42)
h_su = np.random.randn(S, U, N) + 1j * np.random.randn(S, U, N)  # LEO到UAV信道 NX1
H_sR = np.random.randn(S, N, M) + 1j * np.random.randn(S, N, M)  # LEO到RIS信道 NXM
G_Ru = np.random.randn(U, M) + 1j * np.random.randn(U, M)  # RIS到UAV信道 MX1
Phi = np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, M)))  # RIS相移矩阵 M*M

C_su = np.zeros((S, U, N), dtype=complex)  # 等效信道矩阵
for s in range(S):
    for u in range(U):
        C_su[s, u, :] = h_su[s, u, :].T + G_Ru[u, :].T @ Phi @ H_sR[s, :, :].T

# 初始化
w_su = np.zeros((S, U, N), dtype=complex)
for s in range(S):
    for u in range(U):
        w_su[s, u, :] = C_su[s, u, :].conj() / np.linalg.norm(C_su[s, u, :])
    total_power = np.sum([np.linalg.norm(w_su[s, u, :])**2 for u in range(U)])
    w_su[s, :, :] *= np.sqrt(P_s / total_power)
v_u = np.ones(U, dtype=complex)
q_u = np.ones(U)

# SINR和速率计算
def compute_SINR(w_su, C_su):
    SINR = np.zeros(U)
    for u in range(U):
        signal_power = abs(sum(C_su[s, u, :] @ w_su[s, u, :] for s in range(S)))**2
        interference_power = abs(sum(C_su[s, u, :] @ w_su[s, u_prime, :] for s in range(S) for u_prime in range(U) if u_prime != u))**2
        SINR[u] = signal_power / (interference_power + sigma2)
    return SINR

def compute_sum_rate(SINR):
    return np.sum(np.log2(1 + SINR))

# WMMSE算法
def WMMSE_algorithm(w_su, C_su, max_iter=500, tol=1e-5):
    sum_rates = []
    for iteration in range(max_iter):
        # 更新接收滤波器 v_u 
        for u in range(U):
            total_power = sum(abs(C_su[s, u, :] @ w_su[s, u_prime, :])**2 for s in range(S) for u_prime in range(U)) + sigma2
            desired = sum(C_su[s, u, :] @ w_su[s, u, :] for s in range(S))
            v_u[u] = desired / total_power

        # 更新均方误差权重 q_u
        for u in range(U):
            # y_u = sum(C_su[s, u, :] @ w_su[s, u_prime, :] for s in range(S) for u_prime in range(U))
            # e_u = abs(v_u[u] * y_u - 1)**2 + sigma2 * abs(v_u[u])**2
            SINR = compute_SINR(w_su, C_su)
            e_u = 1 / (1 + SINR[u])
            q_u[u] = 1 / e_u

        # 更新 w_{s,u}
        for s in range(S):  # S 为卫星数量
            # 计算 A_s 和 b_su 列表
            A = np.zeros((N, N), dtype=complex)  # N 为天线数
            # b_list = []
            for u in range(U):  # U 为用户数量
                C_u = C_su[s, u, :]  # 信道向量
                A += q_u[u] * np.abs(v_u[u])**2 * np.outer(C_u.conj(), C_u)
                # b_u = q_u[u] * np.conj(v_u[u]) * C_u.conj()
                # b_list.append(b_u)
            
            # 求解 mu
            # mu = solve_mu(A, b_list, P_s)
            mu_low, mu_high = 0, 1e3
            for iter in range(100):
                mu = (mu_low + mu_high) / 2
                power = 0
                for u in range(U):
                    w_temp = np.linalg.solve((A + mu * np.eye(N)), C_su[s, u, :]) * v_u[u] * q_u[u]
                    power += np.linalg.norm(w_temp)**2
                if abs(power - P_s) < tol:
                    break
                if power > P_s:
                    mu_low = mu  # 修正：增大mu以减小功率
                else:
                    mu_high = mu
                # print(f"iter={iter:3d}, mu = {mu:.4f}, power = {power:.4f}")
            # print(f"求解最优mu共迭代{iter:d}次, mu = {mu:.4f}, power = {power:.4f}")
            
            # 更新波束成形向量
            for u in range(U):
                # w_su[s, u, :] = np.linalg.solve(A + mu * np.eye(N), b_list[u])
                w_su[s, u, :] = np.linalg.solve((A + mu * np.eye(N)), C_su[s, u, :]) * v_u[u] * q_u[u]

        # 检查功率
        # for s in range(S):
        #     total_power = np.sum([np.linalg.norm(w_su[s, u, :])**2 for u in range(U)])
        #     print(f"迭代 {iteration+1}, 卫星 {s+1} 的总功率: {total_power:.4f}, 约束: {P_s}")

        # 计算速率
        SINR = compute_SINR(w_su, C_su)
        sum_rate = compute_sum_rate(SINR)
        sum_rates.append(sum_rate)
        print(f"迭代 {iteration+1}, 总速率: {sum_rate:.4f} bps/Hz")
        if iteration > 0 and abs(sum_rates[-1] - sum_rates[-2]) < tol:
            print(f"WMMSE 算法在第 {iteration+1} 次迭代收敛")
            break
    return w_su, sum_rates

# def solve_mu(A, b, P_s, tol=1e-6, max_iter=100):
#     mu_low, mu_high = 0, 1e3
#     for iter in range(max_iter):
#         mu = (mu_low + mu_high) / 2
#         power = 0
#         for u in range(U):
#             w_temp = np.linalg.solve(A + mu * np.eye(N), b[u])
#             power += np.linalg.norm(w_temp)**2
#         if abs(power - P_s) < tol:
#             break
#         if power > P_s:
#             mu_low = mu
#         else:
#             mu_high = mu
#         print(f"iter={iter:3d}, mu = {mu:.4f}, power = {power:.4f}")
#     # print(f"求解最优mu共迭代{iter:d}次, mu = {mu:.4f}, power = {power:.4f}")
#     return mu

# def solve_mu_corrected(A, b_list, P_s, tol=1e-6, max_iter=100):
#     """
#     求解拉格朗日乘子 mu，使得总功率满足约束。
    
#     参数：
#     A : ndarray
#         与卫星 s 相关的矩阵 (N x N)。
#     b_list : list of ndarray
#         包含 U 个用户的 b_su 向量，每个向量形状为 (N,)。
#     P_s : float
#         卫星 s 的功率约束。
#     tol : float, optional
#         容差，默认为 1e-6。
#     max_iter : int, optional
#         最大迭代次数，默认为 100。
    
#     返回：
#     mu : float
#         满足功率约束的拉格朗日乘子。
#     """
#     mu_low = 0
#     mu_high = 1e3  # 初始上限，可根据实际系统调整
    
#     for _ in range(max_iter):
#         mu = (mu_low + mu_high) / 2
#         total_power = 0
#         # 计算所有 U 个波束成形向量的总功率
#         for b in b_list:
#             w_temp = np.linalg.solve(A + mu * np.eye(A.shape[0]), b)
#             total_power += np.linalg.norm(w_temp)**2
        
#         # 检查是否满足容差
#         if abs(total_power - P_s) < tol:
#             return mu
        
#         # 调整二分法范围
#         if total_power > P_s:
#             mu_low = mu
#         else:
#             mu_high = mu
    
#     # 如果达到最大迭代次数，返回当前估计值
#     return mu

# 运行
w_su_opt, sum_rates = WMMSE_algorithm(w_su, C_su)
final_SINR = compute_SINR(w_su_opt, C_su)
print(f"优化后的总速率: {compute_sum_rate(final_SINR):.4f} bps/Hz")
print(f"最终 SINR: {final_SINR}")