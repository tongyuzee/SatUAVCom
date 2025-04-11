import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib import rc
import pylustrator

# 开启pylustrator 
pylustrator.start()

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

def plot_rate_vs_M(file_path='data/rate_vs_M.npz', output_name='rate_vs_M_plot', 
                   color='#1d73b6', marker='o', line_style='-', add_annotations=True,
                   xlabel=r'Number of RIS Elements, $\sqrt{M} $', 
                   ylabel='Sum Rate (bps/Hz)'
                   ):
    """
    读取保存的数据并绘制RIS元素数量与和速率的关系图
    
    参数:
        file_path: 数据文件路径
        output_name: 输出图像的名称
        color: 曲线颜色
        marker: 标记样式
        line_style: 线条样式
        add_annotations: 是否在数据点上添加注释
        xlabel: x轴标签
        ylabel: y轴标签
        fig_width_cm: 图像宽度(厘米)
        fig_height_cm: 图像高度(厘米)
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载数据
    data = np.load(file_path)
    M_values = data['M_values']
    # M_values = M_values**2
    Rate_values = data['Rate_values']
    
    # 设置图像大小（厘米转英寸）
    cm_to_inch = 1/2.54
    fig_width_cm = 15  # 宽度，厘米
    fig_height_cm = 12  # 高度，厘米
    plt.figure(figsize=(fig_width_cm*cm_to_inch, fig_height_cm*cm_to_inch))
    
    # 绘制图像
    plt.plot(M_values, Rate_values, marker=marker, linestyle=line_style, 
             color=color, linewidth=2, markersize=6, 
             markeredgewidth=2)
    
    # 添加标签
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 在每个点上标注具体数值
    if add_annotations:
        for i, (m, r) in enumerate(zip(M_values, Rate_values)):
            plt.annotate(f'{r:.2f}', xy=(m, r), xytext=(0, 8), 
                        textcoords='offset points', ha='center', fontsize=10)
    
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
    
    # 保存图表
    if not os.path.exists('fig'):
        os.makedirs('fig')
    plt.tight_layout()
    plt.savefig(f'fig/{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}.svg', format='svg', bbox_inches='tight') 
    plt.savefig(f'fig/{output_name}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 打印一些统计信息
    print(f"RIS元素数量: {M_values}")
    print(f"对应的和速率: {Rate_values}")
    print(f"最大和速率: {np.max(Rate_values):.4f} 在 M = {M_values[np.argmax(Rate_values)]}")
    
    return M_values, Rate_values

# 使用示例
if __name__ == "__main__":
    plot_rate_vs_M(
        file_path='data/rate_vs_M.npz',
        output_name='RIS_elements_impact',
        color='#1d73b6',
        add_annotations=False
    )