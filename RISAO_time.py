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
        h_tilde = np.zeros((self.U, self.S, self.N), dtype=complex)
        for u in range(self.U):
            for s in range(self.S):
                h_tilde[u, s, :] = self.h_su[u, s, :].T + self.g_Ru[u, :] @ Phi @ self.H_sR[s, :, :].T
                H[s * self.N:(s + 1) * self.N, u] =  h_tilde[u, s, :]
        return H, h_tilde

    def initialize_W(self, H, H_tilde):
        """基于MRT初始化预编码矩阵 W"""
        W = np.zeros((self.S * self.N, self.U), dtype=complex)
        W_su = np.zeros((self.U, self.S, self.N), dtype=complex)
        for u in range(self.U):
            for s in range(self.S):
                W_su[u, s, :] = H_tilde[u, s, :].T / np.linalg.norm(H_tilde[u, s, :], 2)  # 归一化
                W[s * self.N:(s + 1) * self.N, u] = W_su[u, s, :] * np.sqrt(self.P_s / self.S / self.U)  # 分配功率
        # for i in range(self.U):
        #     h_i = H[:, i]  # 第 i 个用户的信道向量
        #     w_i = h_i / np.linalg.norm(h_i, 2)  # 归一化
        #     W[:, i] = w_i * np.sqrt(self.P_s / self.U)  # 分配功率
        return W

    def optimize_W(self, H, W_init):
        """使用WMMSE算法优化预编码矩阵 W"""
        Woptimization = FindW_WMMSE.FindW_WMMSE(self.S, self.U, self.N, self.M, H, W_init, self.P_s, self.sigma2)
        W_opt, rate_w = Woptimization.optimize()
        # Woptimization.plot_rate(rate_w)
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
        # PhiOptimization.plot_rate(rate_phi)
        return theta, rate_phi

    def compute_Sinr_Rsum(self, h_su, H_sR, g_Ru, W_su, theta):
        """计算SINR和和速率"""
        exp_j_theta = np.exp(1j * theta)
        Phi = np.diag(exp_j_theta)  # RIS相位矩阵，size: M x M
        SINR = np.zeros(self.U)
        sigout = np.zeros(self.U, dtype=complex)
        for u in range(self.U):
            signal = 0
            for s in range(self.S):
                equiv_channel = h_su[u, s, :] + g_Ru[u, :] @ Phi @ H_sR[s, :, :].T
                signal += np.vdot(equiv_channel, W_su[u, s, :])
            sigout[u] = signal
            signal_power = np.abs(signal) ** 2
            interference_power = 0
            for u_prime in range(self.U):
                if u_prime != u:
                    interf = 0
                    for s in range(self.S):
                        equiv_channel = h_su[u, s, :] + g_Ru[u, :] @ Phi @ H_sR[s, :, :].T
                        interf += np.vdot(equiv_channel, W_su[u_prime, s, :])
                    interference_power += np.abs(interf) ** 2
            SINR[u] = signal_power / (interference_power + self.sigma2)
        R_sum = np.sum(np.log2(1 + SINR))
        return sigout, SINR, R_sum

    def run_optimization(self, max_iter=1000, tol=1e-3, RandomPhi=False, MRCw=False):
        """执行联合优化过程"""
        for iter in range(max_iter):
            H, H_tilde = self.compute_equivalent_channel()
            if MRCw:  # 如果使用MRC预编码
                W_opt = self.initialize_W(H, H_tilde)
            else:
                W = self.initialize_W(H, H_tilde) if iter == 0 else W_opt
                W_opt, rate_w = self.optimize_W(H, W)
            for u in range(self.U):
                for s in range(self.S):
                    self.W_su[u, s, :] = W_opt[s * self.N:(s + 1) * self.N, u]
            sigout, _, R_w = self.compute_Sinr_Rsum(self.h_su, self.H_sR, self.g_Ru, self.W_su, self.theta)
            self.Rate.append(R_w)
            if self.M > 0 and RandomPhi == False:  # 如果有RIS元素，则优化相位
                theta, rate_phi = self.optimize_theta(self.W_su, R_w)  # R_init=0，与原代码一致
                self.theta = theta
                sigout, _, R_phi = self.compute_Sinr_Rsum(self.h_su, self.H_sR, self.g_Ru, self.W_su, self.theta)
                self.Rate.append(R_phi)
                if MRCw:
                    print(f'迭代次数: {iter:03d}, FindW: MRCw, rate_w={R_w:016.12f}, FindPhi: iter={len(rate_phi):03d}, rate_phi={rate_phi[-1]:016.12f}')
                else:
                    print(f'迭代次数: {iter:03d}, FindW: iter={len(rate_w):03d}, rate_w={rate_w[-1]:016.12f}, FindPhi: iter={len(rate_phi):03d}, rate_phi={rate_phi[-1]:016.12f}')
                if iter > 0 and abs(self.Rate[-1] - self.Rate[-2]) < 1e-4 and abs(self.Rate[-2] - self.Rate[-3]) < 1e-3:
                    break
            else: # 如果没有RIS元素，则只优化W
                # self.theta = np.random.uniform(0, 2 * np.pi, self.M)  # 随机初始化RIS相位
                print(f'迭代次数: {iter:03d}, FindW: iter={len(rate_w):03d}, rate_w={rate_w[-1]:016.12f}')
                if iter > 0 and abs(self.Rate[-1] - self.Rate[-2]) < 1e-4 :
                    break
        # np.save(f'data/AO_rate_1e6.npy', self.Rate)
        return sigout, self.Rate[-1], self.W_su, self.theta

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
    """主函数：设置参数并运行优化，并存储预编码矩阵和相位信息"""
    set_seed(42)
    dt = 3  # 延时
    S = 2  # 卫星数量
    U = 3  # 无人机数量
    N = 4  # 天线数量
    M = 6400  # RIS单元数量
    Rphi = False  # 随机初始化相位
    MRCw = False  # 是否使用MRC预编码
    P_s = 1  # 发射功率 波束形成向量的约束
    PowerdB = 70  # 发射功率(dBm)
    Power = 10 ** (PowerdB / 10) / 1000  # 转换为瓦特
    A = np.sqrt(Power)  # 幅度增益
    gain_factor = 1e4  # 增益因子

    N0 = -85  # dBm/Hz
    B = 1
    N0 = 10 ** (N0 / 10) / 1000 * B  # 噪声功率转换为瓦特

    sigo = []
    Rate_list = []
    Rate_dt_list = []
    
    # 创建存储预编码矩阵和相位的列表
    W_su_all = []  # 存储所有时刻的预编码矩阵
    theta_all = []  # 存储所有时刻的相位
    
    T_list = range(0, 1000, 15)
    time_steps = []  # 记录有效时间步
    time_steps_dt = []
    
    for t in T_list:
        print(f'当前时间：{t}')

        # 生成信道矩阵
        try:
            Sat_UAV_comm = RISSatUAVCom.RISSatUAVCom(t, U, S, N, M)
            h_su, H_sR, g_Ru = Sat_UAV_comm.setup_channel()
        except Exception as e:
            print(f"在时间 t={t} 生成信道发生错误: {e}")
            continue

        if t >= Sat_UAV_comm.TT:
            break

        # 在通信系统中，信号功率通常是∣h^H w∣^2 
        h_su = np.conj(h_su)
        H_sR = np.conj(H_sR)
        g_Ru = np.conj(g_Ru)

        # 信道矩阵放大n倍，噪声功率放大n^2倍，SINR不变，优化结果不变
        h_su = h_su * A * gain_factor 
        H_sR = H_sR * A * gain_factor
        sigma2 = N0 * gain_factor ** 2

        # 实例化并运行优化
        system = RISAlternatingOptimization(S, U, N, M, P_s, sigma2, h_su, H_sR, g_Ru)
        sigout, Rate, W_su_dt, theta_dt = system.run_optimization(RandomPhi=Rphi, MRCw=MRCw)  

        # 存储预编码矩阵和相位
        W_su_all.append(W_su_dt.copy())
        theta_all.append(theta_dt.copy())

        # 生成新的信道矩阵dt
        try:
            Sat_UAV_comm = RISSatUAVCom.RISSatUAVCom(t+dt, U, S, N, M)
            h_su_dt, H_sR_dt, g_Ru_dt = Sat_UAV_comm.setup_channel()
        except Exception as e:
            print(f"在时间 t={t+dt} 生成信道发生错误: {e}")
            continue
            
        # 在通信系统中，信号功率通常是∣h^H w∣^2 
        h_su_dt = np.conj(h_su_dt)
        H_sR_dt = np.conj(H_sR_dt)
        g_Ru_dt = np.conj(g_Ru_dt)

        # 信道矩阵放大n倍，噪声功率放大n^2倍，SINR不变，优化结果不变
        h_su_dt = h_su_dt * A * gain_factor 
        H_sR_dt = H_sR_dt * A * gain_factor
        sigma2 = N0 * gain_factor ** 2
        
        sigout_dt, _, Rate_dt = system.compute_Sinr_Rsum(h_su_dt, H_sR_dt, g_Ru_dt, W_su_dt, theta_dt)

        sigo.append(sigout)
        Rate_list.append(Rate)
        Rate_dt_list.append(Rate_dt)
        time_steps.append(t)
        time_steps_dt.append(t+dt)
        
        print(f'当前时间：{t}, Rate:{Rate:016.12f}, Rate_dt:{Rate_dt:016.12f}')

    # 创建数据目录（如果不存在）
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 保存和速率数据
    np.save(f'data/Whole_Service_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.npy', np.array(Rate_dt_list))
    
    # 保存预编码矩阵和相位数据
    # 将预编码矩阵转换为numpy数组
    W_su_array = np.array(W_su_all)
    theta_array = np.array(theta_all)
    time_array = np.array(time_steps)
    
    # 保存数据到文件
    data_dict = {
        'times': time_array,
        'W_su': W_su_array,
        'theta': theta_array,
        'Rate': np.array(Rate_list),
        'Rate_dt': np.array(Rate_dt_list)
    }
    
    # 使用np.savez保存多个数组到一个文件
    np.savez(f'data/Precoding_Params_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.npz', **data_dict)
    
    # 单独保存预编码矩阵和相位，以防需要单独访问
    np.save(f'data/W_su_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.npy', W_su_array)
    np.save(f'data/theta_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.npy', theta_array)
    
    # 保存系统配置参数，便于后续分析
    config_dict = {
        'S': S,
        'U': U,
        'N': N,
        'M': M,
        'dt': dt,
        'Rphi': Rphi,
        'MRCw': MRCw,
        'PowerdB': PowerdB,
        'gain_factor': gain_factor,
        'N0_dBm': -85
    }
    np.savez(f'data/config_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.npz', **config_dict)

    # 绘制结果
    cm_to_inch = 1/2.54
    fig_width_cm = 15  # 宽度，厘米
    fig_height_cm = 12  # 高度，厘米
    plt.figure(figsize=(fig_width_cm*cm_to_inch, fig_height_cm*cm_to_inch))
    plt.plot(time_steps, Rate_list, 'o-', linewidth=2, markersize=6, label='Optimal Rate (t)', color='#1d73b6')
    plt.plot(time_steps_dt, Rate_dt_list, 's--', linewidth=2, markersize=6, label=f'Delayed Rate (t+{dt})', color='#24a645')
    plt.ylabel('Sum Rate (bps/Hz)', fontsize=14)
    plt.xlabel('Service Time (s)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    
    # 设置框和刻度线 - 类似Matlab的box on
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # 设置刻度线朝内并出现在所有四个边上
    ax.tick_params(axis='both', which='both', direction='in', 
                  top=True, bottom=True, left=True, right=True)
    
    # 添加次刻度线
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # 创建图形目录
    if not os.path.exists('fig'):
        os.makedirs('fig')
        
    # 保存图形
    plt.tight_layout()
    plt.savefig(f'fig/Whole_Service_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'fig/Whole_Service_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'fig/Whole_Service_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n模拟完成。预编码矩阵和相位参数已保存到data目录。")
    print(f"保存的文件:")
    print(f"1. 和速率数据: data/Whole_Service_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.npy")
    print(f"2. 预编码参数: data/Precoding_Params_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.npz")
    print(f"3. 预编码矩阵: data/W_su_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.npy")
    print(f"4. 相位参数: data/theta_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.npy")
    print(f"5. 系统配置: data/config_S{S}_U{U}_N{N}_M{M}_Random{Rphi:d}_MRC{MRCw:d}_dt{dt}.npz")

def analyze_M_impact(time=200, M1_range=None):
    """
    分析固定时刻下和速率随RIS元素个数的变化
    
    参数:
        time: 固定的时间点
        M_range: RIS元素数量范围，默认为[4, 8, 16, 32, 64]
    """
    set_seed(42)  # 保证结果可重现
    
    if M1_range is None:
        M1_range = [4, 8, 16, 32, 64]
    
    S = 2  # 卫星数量
    U = 3  # 无人机数量
    N = 4  # 天线数量
    P_s = 1  # 发射功率 波束形成向量的约束
    PowerdB = 70  # 发射功率(dBm)
    Power = 10 ** (PowerdB / 10) / 1000  # 转换为瓦特
    A = np.sqrt(Power)  # 幅度增益
    # sigma2 = 1e-16  # 噪声方差
    gain_factor = 1e4  # 增益因子
    N0 = -85  # dBm/Hz
    # B = 20e6 # 20MHz
    B = 1
    N0 = 10 ** (N0 / 10) / 1000 * B  # 噪声功率转换为瓦特

    sigoot_vs_M = []
    rate_vs_M = []
    print(f"分析时刻 t={time} 下和速率随RIS元素数量的变化关系")
    
    for M1 in M1_range:
        M = M1 ** 2  # RIS单元数量
        print(f"正在计算 M={M} 的和速率...")
        
        # 生成信道矩阵
        Sat_UAV_comm = RISSatUAVCom.RISSatUAVCom(time, U, S, N, M)
        h_su, H_sR, g_Ru = Sat_UAV_comm.setup_channel()
        
        # 在通信系统中，信号功率通常是∣h^H w∣^2 
        h_su = np.conj(h_su)
        H_sR = np.conj(H_sR)
        g_Ru = np.conj(g_Ru)
        
        # 信道矩阵放大n倍，噪声功率放大n^2倍，SINR不变，优化结果不变
        h_su = h_su * A * gain_factor 
        H_sR = H_sR * A * gain_factor
        sigma2 = N0 * gain_factor ** 2
        # sigma2_scaled = 0.1  # 噪声方差
        
        # 实例化并运行优化
        system = RISAlternatingOptimization(S, U, N, M, P_s, sigma2, h_su, H_sR, g_Ru)
        sigout, rate, _, _ = system.run_optimization()
        # system.plot_results()
        rate_vs_M.append(rate)
        sigoot_vs_M.append(sigout)
    
    # 保存数据
    if not os.path.exists('data'):
        os.makedirs('data')
    results = {'M_values': M1_range, 'Rate_values': rate_vs_M}
    np.savez('data/rate_vs_M.npz', **results)
    
    # 绘制结果
    plt.figure(figsize=(8, 6))
    plt.plot(M1_range, rate_vs_M, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Number of RIS Elements (M)')
    plt.ylabel('Sum Rate (bps/Hz)')
    plt.grid(True)
    # plt.title(f'Sum Rate vs. RIS Elements at t={time}')
    
    # # 在每个点上标注具体数值
    # for i, (m, r) in enumerate(zip(M_range, rate_vs_M)):
    #     plt.annotate(f'{r:.2f}', xy=(m, r), xytext=(0, 10), 
    #                 textcoords='offset points', ha='center')
    
    # 保存图表
    if not os.path.exists('fig'):
        os.makedirs('fig')
    plt.tight_layout()
    plt.savefig('fig/rate_vs_M.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('fig/rate_vs_M.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig/rate_vs_M.svg', format='svg', bbox_inches='tight')
    plt.show()
    return M1_range, rate_vs_M

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
    main_service()
    # analyze_M_impact(time=260, M_range=[16, 64, 256, 1024, 4096])
    # M1_range = range(0, 800, 50)
    # analyze_M_impact(time=150,  M1_range = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
