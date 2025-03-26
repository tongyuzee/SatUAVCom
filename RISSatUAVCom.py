import numpy as np


def aod(A, B, C):
    """计算信号出发角, AB与AC的夹角"""
    AB = np.array(B) - np.array(A)
    AC = np.array(C) - np.array(A)
    dot_product = np.sum(np.multiply(AB, AC), axis=-1, keepdims=True)  # 按行点积
    cos_theta = dot_product / (np.linalg.norm(AB, axis=-1, keepdims=True) * np.linalg.norm(AC, axis=-1, keepdims=True))
    return np.arccos(cos_theta)


def aoa(V):
    """
    计算信号到达角
    返回相对于yOz平面上的方位角和俯仰角

    参数:
    V (numpy array): 信号到达点的坐标

    返回值:
    tuple: 包含方位角和俯仰角的元组，单位为弧度
    """
    V_x, V_y, V_z = np.hsplit(V, 3)
    magnitude = np.linalg.norm(V, axis=-1)
    azimuth = np.arctan2(V_z, V_y)
    elevation = np.arcsin(V_x / magnitude.reshape(-1, 1))
    return azimuth, elevation


class RISSatUAVCom:
    def __init__(self,
                 current_t,
                 U=3,
                 S=2,
                 N=4,
                 M=16,
                 G_S=40,
                 G_R=0,
                 G_T=0,
                 k=30):

        self.current_t = current_t

        self.U = U  # UAV数量(单天线)
        self.S = S  # 卫星数量
        self.N = N  # 卫星天线数量
        self.M = M  # RIS的元素数量
        self.M1 = np.sqrt(self.M)
        self.M2 = np.sqrt(self.M)

        self.c = 3e8  # 光速，单位：米/秒
        self.f_c = 20e9  # 载波频率，单位：赫兹
        self.wavelength = self.c / self.f_c  # 计算波长

        self.G_S = G_S  # 卫星发射天线增益，单位：dBi
        self.G_R = G_R  # RIS增益，单位：dBi
        self.G_T = G_T  # 接收天线增益，单位：dBi
        self.k = 10 ** (k / 10)  # 莱斯因子

        self.h_su = np.zeros((self.U, self.S, self.N), dtype=complex)  # LEO到UAV信道
        self.H_sR = np.zeros((self.S, self.N, self.M), dtype=complex)  # LEO到RIS信道
        self.g_Ru = np.zeros((self.U, self.M), dtype=complex) # RIS到UAV信道

        self.RE = 6371E3  # 地球半径，单位：米
        self.h = 600e3  # 卫星高度，单位：米
        self.D = self.h + self.RE  # 卫星轨道半径，单位：米
        self.v = np.sqrt(3.986e14/self.D)  # 卫星的速度，单位：米/秒
        self.w = self.v / self.D  # 卫星的角速度，单位：弧度/秒
        self.theta0 = 70 * np.pi / 180  # 卫星初始位置，单位：弧度
        self.alpha = 5 * np.pi / 180  # 卫星的角度间隔，单位：弧度

        self.TT = np.round((np.pi - self.theta0 - self.theta0 - self.alpha) / self.w)  # 卫星飞行时间，单位：秒n

        self.l = 10  # RIS与TR之间的水平距离，单位：米
        self.hTR = 100  # TR高度，单位：米
        self.hRIS = 110  # RIS高度，单位：米
        self.delta = self.wavelength / 10  # RIS的元素间距，单位：米

        self.theta = np.zeros(self.S)
        self.pSAT = np.zeros((self.S, 3))
        self.pUAV_initial = np.array([
            [0,-5e3, self.RE + self.hTR], 
            [0, 50-5e3, self.RE + self.hTR], 
            [50, 50-5e3, self.RE + self.hTR]
            ])
        self.pRIS_initial = np.array([50, -5e3, self.RE + self.hRIS])
        self.v_formation = np.array([0, 20, 0])  # 编队速度，单位：m/s，沿 y 轴
        

    def calculate_beta(self, G_X, G_Y, px, py):
        """计算电磁波传播的幅度增益。"""
        G_X = 10 ** (G_X / 10)
        G_Y = 10 ** (G_Y / 10)
        d_XY = np.linalg.norm(np.array(px) - np.array(py), axis=-1)
        return (self.wavelength * np.sqrt(G_X * G_Y)) / (4 * np.pi * d_XY)

    def current_position(self):
        """计算卫星的位置"""
        self.theta = np.arange(self.S)*self.alpha + self.theta0 + self.w * self.current_t
        self.pSAT = np.array([[0, self.D * np.cos(x), self.D * np.sin(x)] for x in self.theta])
        # 更新 UAV 和 RIS 的位置
        self.pUAV = self.pUAV_initial + self.v_formation * self.current_t  # 广播到 (U, 3)
        self.pRIS = self.pRIS_initial + self.v_formation * self.current_t  # (3,)
        return self.pSAT
    
    def compensate_phase_difference(self):
        """
        计算并补偿卫星到UAV之间距离差异引起的相位差异
        返回补偿矩阵以用于预编码
        """
        phase_compensation_su = np.zeros((self.U, self.S), dtype=complex)
        phase_compensation_sR = np.zeros(self.S, dtype=complex)
        
        # 计算每个卫星到每个UAV的距离
        d_su = np.zeros((self.U, self.S))
        d_sR = np.zeros(self.S)
        for s in range(self.S):
            d_sR[s] = np.linalg.norm(self.pRIS - self.pSAT[s])
            for u in range(self.U):
                d_su[u, s] = np.linalg.norm(self.pSAT[s] - self.pUAV[u])
        
        # 计算相位差异并生成补偿因子
        for s in range(self.S):
            phase_compensation_sR[s] = np.exp(-1j * 2 * np.pi * (d_sR[s] - d_sR[0]) / self.wavelength)
            for u in range(self.U):
                # 计算距离差引起的相位差
                phase_diff = 2 * np.pi * (d_su[u, s] - d_su[u, 0]) / self.wavelength
                # 生成补偿因子（共轭形式以抵消相位差）
                phase_compensation_su[u, s] = np.exp(-1j * phase_diff)

        return phase_compensation_su, phase_compensation_sR

    def setup_channel(self):
        """设置信道并计算相关参数。"""
        self.pSAT = self.current_position()
        beta_su = np.zeros((self.U, self.S))  # size:(U, S)
        for u in range(self.U):
            for s in range(self.S):
                beta_su[u, s] = self.calculate_beta(self.G_S, self.G_T, self.pUAV[u], self.pSAT[s]) 
        beta_sR = self.calculate_beta(self.G_S, self.G_R, self.pRIS, self.pSAT)  # size:(S)
        beta_Ru = self.calculate_beta(self.G_T, self.G_R, self.pUAV, self.pRIS)  # size:(U)

        h_su_NLoS = np.random.normal(0, np.sqrt(0.5), (self.U, self.S, self.N)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.U, self.S, self.N))  # LEO到UAV信道
        H_sR_NLoS = np.random.normal(0, np.sqrt(0.5), (self.S, self.N, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.S, self.N, self.M))  # LEO到RIS信道

        aoa_sR_a, aoa_sR_e = aoa(np.array(self.pRIS) - np.array(self.pSAT))  # size:(S)
        aod_su = np.zeros((self.U, self.S))  # size:(U, S)
        for u in range(self.U):
            for s in range(self.S):
                aod_su[u, s] = aod(self.pSAT[s], self.pUAV[u], [0, 0, 0])
        aod_sR = aod(self.pSAT, self.pRIS, [0, 0, 0]) # size:(S)

        aoa_z = self.f_sv(self.M2, self.delta, np.sin(aoa_sR_e) * np.sin(aoa_sR_a))
        aoa_y = self.f_sv(self.M1, self.delta, np.sin(aoa_sR_e) * np.cos(aoa_sR_a))

        delta_phi_su, delta_phi_sR = self.compensate_phase_difference()  # size:(U, S) 用于补偿卫星到UAV之间的相位差

        delta_phi_su = np.ones((self.U, self.S), dtype=complex)
        delta_phi_sR = np.ones(self.S, dtype=complex)

        h_su_LoS = np.zeros((self.U, self.S, self.N), dtype=complex)  # size:(U, S, N)
        for u in range(self.U):
            for s in range(self.S):
                h_su_LoS[u, s] = self.f_sv(self.N, self.wavelength / 2, np.sin(aod_su[u, s])) * delta_phi_su[u, s]
                self.h_su[u, s] = beta_su[u, s] * (np.sqrt(self.k / (1 + self.k)) * h_su_LoS[u, s] + np.sqrt(1 / (1 + self.k)) * h_su_NLoS[u, s])
        
        H_sR_LoS_1 = np.array([np.kron(aoa_z[i], aoa_y[i]) for i in range(self.S)])  # size:(S, M=M2*M1)
        H_sR_LoS_2 = self.f_sv(self.N, self.wavelength / 2, np.sin(aod_sR))  # size:(I, N)
        H_sR_LoS = np.array([H_sR_LoS_2[i].reshape(-1, 1) @ H_sR_LoS_1[i].reshape(1, -1) * delta_phi_sR[i] for i in range(self.S)])  # size:(S, N, M)
        H_sR_tmp = np.sqrt(self.k / (1 + self.k)) * H_sR_LoS + np.sqrt(1 / (1 + self.k)) * H_sR_NLoS
        self.H_sR = np.array([beta_sR[i]*H_sR_tmp[i] for i in range(self.S)])

        for u in range(self.U):
            d_Ru = np.linalg.norm(np.array(self.pUAV[u]) - np.array(self.pRIS))
            self.g_Ru[u] = beta_Ru[u] * np.exp(-1j * 2 * np.pi * d_Ru / self.wavelength) * np.array(np.ones(self.M))

        # sigma = np.array([beta_sR[i] / beta_su[i] * g_Ru * H_sR_LoS_1[i] for i in range(self.S)])
        return self.h_su, self.H_sR, self.g_Ru, delta_phi_su, delta_phi_sR

    def f_sv(self, k, l, gamma):
        """生成导向向量"""
        k_vals = np.arange(int(k))  # 创建从 0 到 k-1 的数组
        const = -1j * 2 * np.pi * l / self.wavelength  # 提取常量部分
        return np.exp(const * k_vals * gamma)  # 使用广播进行矢量化计算
        # return np.array([[np.exp(-1j * 2 * np.pi * l * i * g / self.wavelength) for i in range(int(k))] for g in np.hstack(gamma)])

if __name__ == "__main__":
    Sat_UAV_comm = RISSatUAVCom(10)
    h_su, H_sR, g_Ru = Sat_UAV_comm.setup_channel()
    print(h_su, H_sR, g_Ru)
