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

def plot_algorithm_convergence(wmmse_file='data/FindW_WMMSE.npy', 
                             ga_file='data/FindPhi_GA.npy',
                             output_name='algorithm_convergence'):
    """
    读取WMMSE和GA算法的数据，绘制上下两个子图展示收敛过程（仅和速率曲线）
    
    参数:
        wmmse_file: WMMSE算法数据文件路径
        ga_file: GA算法数据文件路径
        output_name: 输出图像名称
    """
    # 检查文件是否存在
    if not os.path.exists(wmmse_file):
        raise FileNotFoundError(f"文件不存在: {wmmse_file}")
    if not os.path.exists(ga_file):
        raise FileNotFoundError(f"文件不存在: {ga_file}")
    
    # 加载数据
    wmmse_data = np.load(wmmse_file)
    ga_data = np.load(ga_file)
    
    # 设置图像大小（厘米转英寸）
    cm_to_inch = 1/2.54
    fig_width_cm = 15
    fig_height_cm = 12
    
    # 创建上下子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width_cm*cm_to_inch, fig_height_cm*cm_to_inch))
    
    # 指定颜色
    color1 = '#1d73b6'  # 蓝色
    color2 = '#24a645'  # 绿色
    
    # WMMSE子图 - 速率曲线
    ax1.set_xlabel('Iterations', fontsize=14)
    ax1.set_ylabel('Sum Rate (bps/Hz)', fontsize=14)
    ax1.plot(wmmse_data[:21], color=color1, 
           marker='o', linestyle='-', linewidth=2, 
           markersize=6, markeredgewidth=2, label='WMMSE')
    ax1.legend(loc='lower right', fontsize=14)
    
    # GA子图 - 速率曲线
    ax2.set_xlabel('Iterations', fontsize=14)
    ax2.set_ylabel('Sum Rate (bps/Hz)', fontsize=14)
    ax2.plot(range(len(ga_data)), ga_data, color=color2, 
           linestyle='-', linewidth=2, 
           markersize=6, markeredgewidth=2, label='MO')
    ax2.legend(loc='lower right', fontsize=14)
    
    # 设置框和刻度线
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='both', direction='in', 
                      top=True, bottom=True, left=True, right=True)
        
        # 添加次刻度线
        from matplotlib.ticker import AutoMinorLocator
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # # 设置标题
    # ax1.set_title('WMMSE Algorithm Convergence', fontsize=16)
    # ax2.set_title('Gradient Ascent Algorithm Convergence', fontsize=16)
    
    # 设置y轴范围，使其更有可比性
    wmmse_min = min(wmmse_data)
    wmmse_max = max(wmmse_data)
    ga_min = min(ga_data)
    ga_max = max(ga_data)
    
    # 添加一点边距
    wmmse_padding = (wmmse_max - wmmse_min) * 0.1
    ga_padding = (ga_max - ga_min) * 0.1
    
    ax1.set_ylim([wmmse_min - wmmse_padding, wmmse_max + wmmse_padding])
    ax2.set_ylim([ga_min - ga_padding, ga_max + ga_padding])
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3)  # 调整子图间距
    
    # 保存图表
    if not os.path.exists('fig'):
        os.makedirs('fig')
    plt.savefig(f'fig/{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}.svg', format='svg', bbox_inches='tight') 
    plt.savefig(f'fig/{output_name}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 输出统计信息
    print("WMMSE算法统计信息:")
    print(f"初始和速率: {wmmse_data[0]:.4f}")
    print(f"最终和速率: {wmmse_data[-1]:.4f}")
    print(f"提升: {wmmse_data[-1] - wmmse_data[0]:.4f} ({(wmmse_data[-1] - wmmse_data[0])/wmmse_data[0]*100:.2f}%)")
    print(f"迭代次数: {len(wmmse_data)}")
    
    print("\nGA算法统计信息:")
    print(f"初始和速率: {ga_data[0]:.4f}")
    print(f"最终和速率: {ga_data[-1]:.4f}")
    print(f"提升: {ga_data[-1] - ga_data[0]:.4f} ({(ga_data[-1] - ga_data[0])/ga_data[0]*100:.2f}%)")
    print(f"迭代次数: {len(ga_data)}")

if __name__ == "__main__":
    plot_algorithm_convergence(
        wmmse_file='data/FindW_WMMSE.npy',
        ga_file='data/FindPhi_GA.npy',
        output_name='WMMSE_GA_convergence_simple'
    )