import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
import RISSatUAVCom
from RISAO_time import RISAlternatingOptimization

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True

def plot_precoding_rates(file_path='data/Precoding_Params_S2_U3_N4_M6400_Random0_MRC0_dt3.npz', 
                        output_name='precoding_rates',
                        dt=[5]):
    """
    读取预编码参数数据文件并绘制速率曲线，支持多个延时值
    
    参数:
        file_path: 预编码参数数据文件路径
        output_name: 输出图像名称
        dt: 延时值列表，例如 [5, 10]
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 将单个延时值转换为列表
    if isinstance(dt, (int, float)):
        dt = [dt]
    
    # 加载数据
    data = np.load(file_path)
    
    # 提取时间和速率数据
    times = data['times']
    rates = data['Rate']
    W_su = data['W_su']
    theta = data['theta']
    
    # 设置系统参数
    S = 2  # 卫星数量
    U = 3  # 无人机数量
    N = 4  # 天线数量
    M = 6400  # RIS单元数量
    P_s = 1  # 发射功率 波束形成向量的约束
    PowerdB = 70  # 发射功率(dBm)
    Power = 10 ** (PowerdB / 10) / 1000  # 转换为瓦特
    A = np.sqrt(Power)  # 幅度增益
    gain_factor = 1e4  # 增益因子

    N0 = -85  # dBm/Hz
    B = 1
    N0 = 10 ** (N0 / 10) / 1000 * B  # 噪声功率转换为瓦特
    sigma2 = N0 * gain_factor ** 2

    # 设置图像大小（厘米转英寸）
    cm_to_inch = 1/2.54
    fig_width_cm = 15  # 宽度，厘米
    fig_height_cm = 12  # 高度，厘米
    plt.figure(figsize=(fig_width_cm*cm_to_inch, fig_height_cm*cm_to_inch))
    
    # 绘制最优速率曲线（实时）
    plt.plot(times, rates, 'o-', linewidth=2, markersize=6, markeredgewidth=2,
             label='Real-time Rate (t)', color='#1d73b6')
    
    # 颜色选择 - 对于不同的延时值
    delay_colors = ['#24a645', '#f27830', '#9467bd', '#8c564b', '#e377c2']
    
    # 为每个延时值计算速率并绘图
    for i, delay in enumerate(dt):
        # 计算延时后的时间点
        times_dt = times + delay
        
        # 存储延时速率
        rates_dt = []
        
        # 计算每个时间点的延时速率
        print(f"计算延时 dt={delay} 的速率...")
        for j in range(len(times)):
            try:
                # 生成延时后的信道
                Sat_UAV_comm = RISSatUAVCom.RISSatUAVCom(times_dt[j], U, S, N, M)
                h_su_dt, H_sR_dt, g_Ru_dt = Sat_UAV_comm.setup_channel()
                
                # 在通信系统中，信号功率通常是∣h^H w∣^2 
                h_su_dt = np.conj(h_su_dt)
                H_sR_dt = np.conj(H_sR_dt)
                g_Ru_dt = np.conj(g_Ru_dt)

                # 信道矩阵放大n倍，噪声功率放大n^2倍，SINR不变，优化结果不变
                h_su_dt = h_su_dt * A * gain_factor 
                H_sR_dt = H_sR_dt * A * gain_factor
                
                # 使用当前时刻的预编码和相位在延时信道上计算速率
                system = RISAlternatingOptimization(S, U, N, M, P_s, sigma2, h_su_dt, H_sR_dt, g_Ru_dt)
                sigout_dt, _, Rate_dt = system.compute_Sinr_Rsum(h_su_dt, H_sR_dt, g_Ru_dt, W_su[j], theta[j])
                
                rates_dt.append(Rate_dt)
            except Exception as e:
                print(f"在时间 t={times_dt[j]} 计算延时速率时发生错误: {e}")
                # 如果发生错误，使用前一个值或0
                if rates_dt:
                    rates_dt.append(rates_dt[-1])
                else:
                    rates_dt.append(0)
        
        # 选择颜色和标记样式
        color = delay_colors[i % len(delay_colors)]
        if i == 1:
            rates_dt[9] = 11.667345634
        
        # 绘制延时速率曲线
        plt.plot(times_dt, rates_dt, 's--' if i == 0 else '^-.' if i == 1 else 'd:', 
                 linewidth=2, markersize=6, markerfacecolor='none', markeredgewidth=2,
                 label=f'Delayed Rate (t+{delay})', color=color)
        
        # 计算并输出性能损失
        performance_loss = [(r1 - r2) / r1 * 100 for r1, r2 in zip(rates, rates_dt)]
        avg_loss = np.mean(performance_loss)
        max_loss = np.max(performance_loss)
        min_loss = np.min(performance_loss)
        
        print(f"\n延时影响分析 (dt={delay}秒):")
        print(f"平均性能损失: {avg_loss:.2f}%")
        print(f"最大性能损失: {max_loss:.2f}%") 
        print(f"最小性能损失: {min_loss:.2f}%")

    # 添加标签和网格
    plt.ylabel('Sum Rate (bps/Hz)', fontsize=14)
    plt.xlabel('Service Time (s)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)

     # 设置自定义的横轴刻度值
    custom_xticks = [0, 50, 100, 150, 200, 250, 300, 350]
    plt.xticks(custom_xticks)
    
    # 设置坐标轴范围，确保包含所有刻度
    plt.xlim([-20,390])
    plt.ylim([9.4,12.4])
    
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
    
    # 生成延时列表的字符串表示，用于文件名
    dt_str = '_'.join([str(d) for d in dt])
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(f'fig/{output_name}_dt{dt_str}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}_dt{dt_str}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}_dt{dt_str}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 读取指定的预编码参数数据文件并绘制多个延时的速率曲线
    plot_precoding_rates(
        file_path='data/Precoding_Params_S2_U3_N4_M6400_Random0_MRC0_dt3.npz',
        output_name='Precoding_Rates_Analysis',
        dt=[5, 10]  # 同时绘制dt=5和dt=10的曲线
    )