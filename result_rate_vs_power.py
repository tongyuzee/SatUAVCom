import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True

def plot_power_rate_curves(file_path='data/rate_vs_power_N4_64_t150.npz', 
                          output_name='power_rate_comparison',
                          colors=None,
                          markers=None,
                          add_annotations=False):
    """
    读取保存的功率-速率数据并绘制曲线
    
    参数:
        file_path: 数据文件路径
        output_name: 输出图像名称
        colors: 颜色列表，若为None则使用默认值
        markers: 标记样式列表，若为None则使用默认值
        add_annotations: 是否在数据点上添加注释
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载数据
    data = np.load(file_path)
    
    # 提取功率范围
    PowerdB_range = data['PowerdB_range']
    
    # 提取不同天线数量的速率数据
    N_values = []
    rate_values = []
    
    # 寻找包含'Rate_N'的键
    for key in data.keys():
        if key.startswith('Rate_N'):
            # 从键名提取N值
            N = int(key[6:])
            N_values.append(N)
            rate_values.append(data[key])
    
    # 确保N值按升序排序
    indices = np.argsort(N_values)
    N_values = [N_values[i] for i in indices]
    rate_values = [rate_values[i] for i in indices]
    
    # 设置默认颜色和标记
    if colors is None:
        colors = ['#1d73b6', '#24a645', '#f27830', '#9467bd', '#8c564b']
    
    if markers is None:
        markers = ['o', 's', '^', 'D', 'x']
    
    # 设置图像大小（厘米转英寸）
    cm_to_inch = 1/2.54
    fig_width_cm = 15  # 宽度，厘米
    fig_height_cm = 12  # 高度，厘米
    plt.figure(figsize=(fig_width_cm*cm_to_inch, fig_height_cm*cm_to_inch))
    
    # 绘制各条曲线
    for i, (N, rates) in enumerate(zip(N_values, rate_values)):
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        
        plt.plot(PowerdB_range, rates, marker=markers[marker_idx], linestyle='-', 
                 color=colors[color_idx], linewidth=2, markersize=6, markeredgewidth=2, markerfacecolor='none',
                 label=f'N = {N}')
        
        # 添加数据点注释
        if add_annotations:
            for j, (p, r) in enumerate(zip(PowerdB_range, rates)):
                if j % 2 == 0:  # 可以选择只标注部分点，避免拥挤
                    plt.annotate(f'{r:.2f}', xy=(p, r), xytext=(0, 5), 
                                textcoords='offset points', ha='center', fontsize=8)
    
    # 添加标签和网格
    plt.xlabel('Transmit Power (dBm)', fontsize=14)
    plt.ylabel('Sum Rate (bps/Hz)', fontsize=14)
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
    
    # # 自动设置Y轴范围，让曲线更清晰
    # all_rates = np.concatenate(rate_values)
    # y_min = np.min(all_rates) * 0.95  # 留出5%的空间
    # y_max = np.max(all_rates) * 1.05
    plt.xlim([-4, 94])
    plt.ylim([-1, 23])
    plt.xticks(range(0,100,10))
    
    # 保存图像
    if not os.path.exists('fig'):
        os.makedirs('fig')
    
    plt.tight_layout()
    plt.savefig(f'fig/{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出统计信息
    print("功率-速率分析结果:")
    for i, N in enumerate(N_values):
        rates = rate_values[i]
        
        # 计算功率增益
        min_rate = rates[0]
        max_rate = rates[-1]
        gain = (max_rate - min_rate) / min_rate * 100
        
        print(f"\nN = {N} 天线:")
        print(f"最低功率 ({PowerdB_range[0]} dBm) 速率: {min_rate:.4f} bps/Hz")
        print(f"最高功率 ({PowerdB_range[-1]} dBm) 速率: {max_rate:.4f} bps/Hz")
        print(f"速率提升: {max_rate - min_rate:.4f} bps/Hz ({gain:.2f}%)")
    
    # 比较不同天线数量的性能
    if len(N_values) > 1:
        print("\n天线数量影响分析:")
        for i in range(len(PowerdB_range)):
            power = PowerdB_range[i]
            rates_at_power = [rate_values[j][i] for j in range(len(N_values))]
            min_rate = min(rates_at_power)
            max_rate = max(rates_at_power)
            gain = (max_rate - min_rate) / min_rate * 100
            
            print(f"在功率 {power} dBm 下:")
            print(f"  天线数量从 {min(N_values)} 增加到 {max(N_values)} 的速率提升: {max_rate - min_rate:.4f} bps/Hz ({gain:.2f}%)")
    
    return PowerdB_range, N_values, rate_values

if __name__ == "__main__":
    # 使用示例
    PowerdB_range, N_values, rate_values = plot_power_rate_curves(
        file_path='data/rate_vs_power_N4_64_t150.npz',
        output_name='power_rate_comparison'
    )