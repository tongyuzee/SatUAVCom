import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import RISSatUAVCom
import FindW_WMMSE
import FindPhi_GradientAscent

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True

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
        self.Rate = []  # 存储和速率

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
        PhiOptimization = FindPhi_GradientAscent.FindPhi_GA(
            self.S, self.U, self.N, self.M, h_su_t, H_sR_t, g_Ru_t, W_su_t, theta_t, R_init, self.sigma2
        )
        theta, rate_phi = PhiOptimization.optimize_theta()
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

    def run_optimization(self, max_iter=1000, tol=1e-3):
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
            theta, rate_phi = self.optimize_theta(self.W_su, R_w)  # R_init=0，与原代码一致
            self.theta = theta
            sigout_phi, _, R_phi = self.compute_Sinr_Rsum(self.W_su)
            self.Rate.append(R_phi)
            print(f'迭代次数: {iter:03d}, FindW: iter={len(rate_w):03d}, rate_w={rate_w[-1]:016.12f}, FindPhi: iter={len(rate_phi):03d}, rate_phi={rate_phi[-1]:016.12f}')
            if iter > 0 and abs(self.Rate[-1] - self.Rate[-2]) < 1e-4 and abs(self.Rate[-2] - self.Rate[-3]) < 1e-3:
                break
        return sigout_phi, self.Rate[-1], self.W_su, self.theta

    def plot_results(self):
        """绘制和速率曲线和误差曲线在同一张图上，使用双纵轴"""
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # 计算误差（相邻迭代之间的和速率变化）
        errors = []
        for i in range(1, len(self.Rate)):
            errors.append(abs(self.Rate[i] - self.Rate[i-1]))
        
        # 和速率曲线 (左纵轴)
        color = 'tab:blue'
        ax1.set_xlabel('Iterations', fontsize=14)
        ax1.set_ylabel('Sum Rate (bps/Hz)', color=color, fontsize=14)
        line1 = ax1.plot(range(len(self.Rate)), self.Rate, color=color, 
                        marker='o', linestyle='-', linewidth=2, 
                        label='Sum Rate')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 创建右侧纵轴用于误差曲线
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Error (absolute difference)', color=color, fontsize=14)
        ax2.set_yscale('log')  # 使用对数坐标
        line2 = ax2.plot(range(1, len(self.Rate)), errors, color=color, 
                        marker='s', linestyle='--', linewidth=2,
                        label='Error')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 添加图例 - 合并两个轴的图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(0.75, 0.8), fontsize=12)
        
        # # 添加网格线 (仅适用于左轴)
        # ax1.grid(True)
        # plt.title('Sum Rate and Error vs. Iterations', fontsize=16)
        plt.tight_layout()
        
        # 保存图表
        if not os.path.exists('fig'):
            os.makedirs('fig')
        plt.savefig('fig/AO_with_errors.pdf', format='pdf', bbox_inches='tight')
        plt.savefig('fig/AO_with_errors.svg', format='svg', bbox_inches='tight')
        plt.savefig('fig/AO_with_errors.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # # 也保存原始的单轴图表
        # plt.figure(figsize=(8, 6))
        # plt.plot(self.Rate, marker='o')
        # plt.ylabel('Sum Rate (bps/Hz)', fontsize=14)
        # plt.xlabel('Iterations', fontsize=14)
        # plt.grid(True)
        # plt.savefig('fig/AO.pdf', format='pdf', bbox_inches='tight')
        # plt.savefig('fig/AO.svg', format='svg', bbox_inches='tight')
        # plt.savefig('fig/AO.png', dpi=300, bbox_inches='tight')
        # plt.close()

def main_service():
    """主函数：设置参数并运行优化"""
    set_seed(42)
    S = 2  # 卫星数量
    U = 3  # 无人机数量
    N = 4  # 天线数量
    M = 16  # RIS单元数量
    P_s = 1  # 发射功率
    sigma2 = 1e-16  # 噪声方差
    gain_factor = 1e8  # 增益因子


    sigo = []
    Rate_list = []
    T_list = range(0, 1000, 10)
    for t in T_list:
        print(f'当前时间：{t}')

        if t == 160:
            xxx = 10  # 用于调试
            pass # 用于调试

        # 生成信道矩阵
        Sat_UAV_comm = RISSatUAVCom.RISSatUAVCom(t, U, S, N, M)
        h_su, H_sR, g_Ru= Sat_UAV_comm.setup_channel()

        # 在通信系统中，信号功率通常是∣h^H w∣^2 
        h_su = np.conj(h_su)
        H_sR = np.conj(H_sR)
        g_Ru = np.conj(g_Ru)

        # 信道矩阵放大n倍，噪声功率放大n^2倍，SINR不变，优化结果不变。
        h_su = h_su * gain_factor
        H_sR = H_sR * gain_factor
        sigma2 = 1e-16  # 噪声方差
        sigma2 = sigma2 * gain_factor ** 2

        # 实例化并运行优化
        system = RISAlternatingOptimization(S, U, N, M, P_s, sigma2, h_su, H_sR, g_Ru)
        sigout, Rate, _, _ = system.run_optimization()
        # system.plot_results()
        # sigout, Rate, _, _ = run(t, U, S, N, M, P_s, gain_factor)
        sigo.append(sigout)
        Rate_list.append(Rate)

        if t >= Sat_UAV_comm.TT:
            break
    if not os.path.exists('data'):
            os.makedirs('data')
    # 保存 Rate_list 数据
    np.save('data/Rate_list2.npy', np.array(Rate_list))

    plt.figure(figsize=(8, 6))
    plt.plot(T_list[0:len(Rate_list)], Rate_list)
    plt.ylabel('Sum Rate')
    plt.xlabel('Service time')
    plt.grid(True)
    if not os.path.exists('fig'):
            os.makedirs('fig')
    plt.savefig('fig/Whole_Service.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('fig/Whole_Service.svg', format='svg', bbox_inches='tight')
    plt.show()


def analyze_M_impact(time=200, M_range=None):
    """
    分析固定时刻下和速率随RIS元素个数的变化
    
    参数:
        time: 固定的时间点
        M_range: RIS元素数量范围，默认为[4, 8, 16, 32, 64]
    """
    set_seed(42)  # 保证结果可重现
    
    if M_range is None:
        M_range = [4, 8, 16, 32, 64]
    
    S = 2  # 卫星数量
    U = 3  # 无人机数量
    N = 4  # 天线数量
    P_s = 1  # 发射功率
    sigma2 = 1e-16  # 噪声方差
    gain_factor = 1e8  # 增益因子
    
    sigoot_vs_M = []
    rate_vs_M = []
    print(f"分析时刻 t={time} 下和速率随RIS元素数量的变化关系")
    
    for M in M_range:
        print(f"正在计算 M={M} 的和速率...")
        
        # 生成信道矩阵
        Sat_UAV_comm = RISSatUAVCom.RISSatUAVCom(time, U, S, N, M)
        h_su, H_sR, g_Ru = Sat_UAV_comm.setup_channel()
        
        # 在通信系统中，信号功率通常是∣h^H w∣^2 
        h_su = np.conj(h_su)
        H_sR = np.conj(H_sR)
        g_Ru = np.conj(g_Ru)
        
        # 信道矩阵放大n倍，噪声功率放大n^2倍，SINR不变，优化结果不变
        h_su = h_su * gain_factor
        H_sR = H_sR * gain_factor
        sigma2_scaled = 1e-16 * gain_factor ** 2
        
        # 实例化并运行优化
        system = RISAlternatingOptimization(S, U, N, M, P_s, sigma2_scaled, h_su, H_sR, g_Ru)
        sigout, rate, _, _ = system.run_optimization()
        system.plot_results()
        rate_vs_M.append(rate)
        sigoot_vs_M.append(sigout)
    
    # 保存数据
    if not os.path.exists('data'):
        os.makedirs('data')
    results = {'M_values': M_range, 'Rate_values': rate_vs_M}
    np.savez('data/rate_vs_M.npz', **results)
    
    # 绘制结果
    plt.figure(figsize=(8, 6))
    plt.plot(M_range, rate_vs_M, 'o-', linewidth=2, markersize=10)
    plt.xlabel('Number of RIS Elements (M)')
    plt.ylabel('Sum Rate (bps/Hz)')
    plt.grid(True)
    plt.title(f'Sum Rate vs. RIS Elements at t={time}')
    
    # 在每个点上标注具体数值
    for i, (m, r) in enumerate(zip(M_range, rate_vs_M)):
        plt.annotate(f'{r:.2f}', xy=(m, r), xytext=(0, 10), 
                    textcoords='offset points', ha='center')
    
    # 保存图表
    if not os.path.exists('fig'):
        os.makedirs('fig')
    plt.tight_layout()
    plt.savefig('fig/rate_vs_M.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('fig/rate_vs_M.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig/rate_vs_M.svg', format='svg', bbox_inches='tight')
    plt.show()
    return M_range, rate_vs_M

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
    # main_service()
    # analyze_M_impact(time=260, M_range=[16, 64, 256, 1024, 4096])
    analyze_M_impact(time=260, M_range=[1024])