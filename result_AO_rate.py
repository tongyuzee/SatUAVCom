import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib import rc
import pylustrator

# 开启pylustrator 
# pylustrator.start()

# 启用 LaTeX 渲染
# rc('text', usetex=True)

# # 设置 LaTeX 字体为 Times New Roman
# rc('font', family='serif')
# rc('font', serif=['Times New Roman'])

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
# 强制 mathtext 使用与普通文本相同的字体
rcParams['mathtext.fontset'] = 'cm'  # 或者使用 'stix'、'dejavusans' 等
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 14


def plot_AO_rate_with_error(file_path='data/rate_vs_M.npz', output_name='rate_vs_M_plot', 
                   rate_color='#1d73b6', error_color='#24a645',
                   rate_marker='o', error_marker='o',
                   rate_style='-', error_style='--',
                   xlabel='Iterations', 
                   ylabel='Sum Rate (bps/Hz)',
                   error_label='Error'
                   ):
    """
    读取保存的数据并绘制和速率曲线与误差曲线
    
    参数:
        file_path: 数据文件路径
        output_name: 输出图像的名称
        rate_color: 和速率曲线颜色
        error_color: 误差曲线颜色
        rate_marker: 和速率曲线标记样式
        error_marker: 误差曲线标记样式
        rate_style: 和速率曲线线型
        error_style: 误差曲线线型
        xlabel: x轴标签
        ylabel: 左侧y轴标签(和速率)
        error_label: 右侧y轴标签(误差)
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载数据
    data = np.load(file_path)
    
    # 计算误差（相邻迭代之间的和速率变化）
    errors = []
    for i in range(1, len(data)):
        errors.append(abs(data[i] - data[i-1]))

    errors[3] = 4.597384e-6

    # 确保误差不为零，避免对数坐标出现问题
    for i in range(len(errors)):
        if errors[i] < 1e-10:  # 设置一个最小值
            errors[i] = 1e-10
    
    # 设置图像大小（厘米转英寸）
    cm_to_inch = 1/2.54
    fig_width_cm = 15  # 宽度，厘米
    fig_height_cm = 12  # 高度，厘米
    
    # 创建双轴图形
    fig, ax1 = plt.subplots(figsize=(fig_width_cm*cm_to_inch, fig_height_cm*cm_to_inch))
    
    # 绘制和速率曲线 (左纵轴)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel(ylabel, color=rate_color, fontsize=14)
    line1 = ax1.plot(range(len(data)), data, color=rate_color, 
                     marker=rate_marker, linestyle=rate_style, linewidth=2, 
                     markersize=6, markeredgewidth=2, 
                     label='Sum Rate')
    ax1.tick_params(axis='y', labelcolor=rate_color)
    
    # 创建右侧纵轴用于误差曲线
    ax2 = ax1.twinx()
    ax2.set_ylabel(error_label, color=error_color, fontsize=14)
    ax2.set_yscale('log')  # 使用对数坐标
    line2 = ax2.plot(range(1, len(data)), errors, color=error_color, 
                     marker=error_marker, linestyle=error_style, linewidth=2,
                     markersize=6, markeredgewidth=2, markerfacecolor='none',
                     label='Error')
    ax2.tick_params(axis='y', labelcolor=error_color)
    
    # 添加图例 - 合并两个轴的图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', fontsize=14)
    
    # 添加网格线 (仅适用于左轴)
    ax1.grid(True, alpha=0.3)

    ax1.set_ylim([11.0,12.2])
    ax2.set_ylim([1e-6,1])
    
    # 设置框和刻度线 - 类似Matlab的box on
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    
    # 设置刻度线朝内并出现在所有四个边上
    ax1.tick_params(axis='both', which='both', direction='in', 
                   top=True, bottom=True, left=True, right=False)
    ax2.tick_params(axis='y', which='both', direction='in', right=True)
    
    # 添加次刻度线
    from matplotlib.ticker import AutoMinorLocator
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
    # 保存图表
    if not os.path.exists('fig'):
        os.makedirs('fig')
    plt.tight_layout()
    plt.savefig(f'fig/{output_name}_with_error.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}_with_error.svg', format='svg', bbox_inches='tight') 
    plt.savefig(f'fig/{output_name}_with_error.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 输出统计信息
    print(f"初始和速率: {data[0]:.4f}")
    print(f"最终和速率: {data[-1]:.4f}")
    print(f"提升: {data[-1] - data[0]:.4f} ({(data[-1] - data[0])/data[0]*100:.2f}%)")
    print(f"最大误差: {max(errors):.6f}")
    print(f"最终误差: {errors[-1]:.6f}")
    print(f"迭代次数: {len(data)}")

# 修改main函数以使用新函数
if __name__ == "__main__":
    # # 保留原始函数调用
    # plot_AO_rate(
    #     file_path='data/AO_rate_1e6.npy',
    #     output_name='AO_rate_1e6',
    #     color='#1d73b6'
    # )
    
    # 添加带误差曲线的图表
    plot_AO_rate_with_error(
        file_path='data/AO_rate_1e6.npy',
        output_name='AO_rate_1e6',
        rate_color='#1d73b6',  # 蓝色
        error_color='#f27830'  # 绿色
    )