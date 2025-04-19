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

def plot_rate_vs_M_comparison(file_path1='data/rate_vs_M_Random0.npz', 
                             file_path2='data/rate_vs_M_Random1.npz',
                             output_name='rate_vs_M_comparison', 
                             color1='#1d73b6', color2='#24a645',
                             marker1='o', marker2='o',
                             line_style1='-', line_style2='--',
                             add_annotations=False,
                             xlabel=r'Number of RIS Elements $\sqrt{M}$', 
                             ylabel='Sum Rate (bps/Hz)',
                             label1='Optimized Phase', label2='Random Phase'):
    """
    读取两组数据并在同一张图上绘制RIS元素数量与和速率的关系图
    
    参数:
        file_path1: 第一组数据文件路径
        file_path2: 第二组数据文件路径
        output_name: 输出图像的名称
        color1, color2: 两条曲线的颜色
        marker1, marker2: 两条曲线的标记样式
        line_style1, line_style2: 两条曲线的线型
        add_annotations: 是否在数据点上添加注释
        xlabel, ylabel: 坐标轴标签
        label1, label2: 两条曲线的图例标签
    """
    # 检查文件是否存在
    if not os.path.exists(file_path1):
        raise FileNotFoundError(f"文件不存在: {file_path1}")
    if not os.path.exists(file_path2):
        raise FileNotFoundError(f"文件不存在: {file_path2}")
    
    # 加载数据
    data1 = np.load(file_path1)
    data2 = np.load(file_path2)

    M_values1 = data1['M_values']
    Rate_values1 = data1['Rate_values']
    
    M_values2 = data2['M_values']
    Rate_values2 = data2['Rate_values']

    # 无RIS的数据
    M_values3 = M_values1
    Rate_values3 = np.ones(len(Rate_values1)) * Rate_values1[0]

    # 设置图像大小（厘米转英寸）
    cm_to_inch = 1/2.54
    fig_width_cm = 15  # 宽度，厘米
    fig_height_cm = 12.1  # 高度，厘米
    plt.figure(figsize=(fig_width_cm*cm_to_inch, fig_height_cm*cm_to_inch))
    
    # 绘制第一条曲线
    plt.plot(M_values1, Rate_values1, marker=marker1, linestyle=line_style1, 
             color=color1, linewidth=2, markersize=6, markeredgewidth=2,
             label=label1)
    
    # 绘制第二条曲线
    plt.plot(M_values2, Rate_values2, marker=marker2, linestyle=line_style2, 
             color=color2, linewidth=2, markersize=6, markeredgewidth=2,
             markerfacecolor='none', label=label2)
    
    # 绘制第三条曲线
    plt.plot(M_values3, Rate_values3, linestyle=':', 
             color='#f27830', linewidth=2, 
             label='Without RIS')
    
    # 添加标签
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, loc='best')
    
    # # 在每个点上标注具体数值
    # if add_annotations:
    #     for i, (m, r) in enumerate(zip(M_values1, Rate_values1)):
    #         plt.annotate(f'{r:.2f}', xy=(m, r), xytext=(0, 8), 
    #                     textcoords='offset points', ha='center', fontsize=10)
    #     for i, (m, r) in enumerate(zip(M_values2, Rate_values2)):
    #         plt.annotate(f'{r:.2f}', xy=(m, r), xytext=(0, -15), 
    #                     textcoords='offset points', ha='center', fontsize=10)
    
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

    ax.set_xlim([-5, 105])
    ax.set_ylim([11.4, 12.5])
    
    # 保存图表
    if not os.path.exists('fig'):
        os.makedirs('fig')
    plt.tight_layout()
    plt.savefig(f'fig/{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}.svg', format='svg', bbox_inches='tight') 
    plt.savefig(f'fig/{output_name}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 打印一些统计信息
    print(f"优化相位:")
    print(f"RIS元素数量: {M_values1}")
    print(f"对应的和速率: {Rate_values1}")
    print(f"最大和速率: {np.max(Rate_values1):.4f} 在 M = {M_values1[np.argmax(Rate_values1)]}")
    
    print(f"\n随机相位:")
    print(f"RIS元素数量: {M_values2}")
    print(f"对应的和速率: {Rate_values2}")
    print(f"最大和速率: {np.max(Rate_values2):.4f} 在 M = {M_values2[np.argmax(Rate_values2)]}")
    
    # 计算两组数据的性能提升
    if len(M_values1) == len(M_values2) and np.array_equal(M_values1, M_values2):
        improvements = []
        improvement_percentages = []
        
        for r1, r2 in zip(Rate_values1, Rate_values2):
            imp = r1 - r2
            imp_percent = (imp / r2) * 100 if r2 != 0 else float('inf')
            improvements.append(imp)
            improvement_percentages.append(imp_percent)
        
        print("\n优化相位vs随机相位性能提升:")
        for m, imp, imp_percent in zip(M_values1, improvements, improvement_percentages):
            print(f"M = {m}: 提升 {imp:.4f} bps/Hz ({imp_percent:.2f}%)")
    
    return M_values1, Rate_values1, M_values2, Rate_values2

# 使用示例
if __name__ == "__main__":
    plot_rate_vs_M_comparison(
        file_path1='data/rate_vs_M.npz',  # 优化相位
        file_path2='data/rate_vs_M_Random1.npz',  # 随机相位
        output_name='RIS_elements_impact_comparison',
        color1='#1d73b6',  # 蓝色
        color2='#24a645',  # 绿色
        label1='RIS with optimised-phase elements',
        label2='RIS with random-phase elements',
        add_annotations=False
    )