import numpy as np
import matplotlib.pyplot as plt
import os

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True

def compare_rates_dual_axis(file1, label1, file2, label2, time_interval=10, max_time=None, 
                           output_name="rate_comparison", color1='#1d73b6', color2='#24a645',
                           style1='-', style2='--', marker1='o', marker2='s', ylim1=None, ylim2=None):
    """
    读取两个速率数据文件并在双纵轴上比较
    
    参数:
        file1, file2: 要比较的数据文件路径
        label1, label2: 图例标签
        time_interval: 时间采样间隔
        max_time: 最大显示时间 (可选)
        output_name: 输出文件名前缀
        color1, color2: 两条曲线的颜色
        style1, style2: 两条曲线的线型
        marker1, marker2: 两条曲线的标记
    """
    # 检查文件是否存在
    if not os.path.exists(file1):
        raise FileNotFoundError(f"文件不存在: {file1}")
    if not os.path.exists(file2):
        raise FileNotFoundError(f"文件不存在: {file2}")
    
    # 读取数据
    rate_list1 = np.load(file1)
    rate_list2 = np.load(file2)
    
    # 生成时间列表
    T_list1 = np.arange(0, len(rate_list1) * time_interval, time_interval)
    T_list2 = np.arange(0, len(rate_list2) * time_interval, time_interval)
    
    # 如果指定了最大时间，进行截断
    if max_time:
        idx1 = np.where(T_list1 <= max_time)[0]
        idx2 = np.where(T_list2 <= max_time)[0]
        T_list1 = T_list1[idx1]
        T_list2 = T_list2[idx2]
        rate_list1 = rate_list1[idx1]
        rate_list2 = rate_list2[idx2]

    cm_to_inch = 1/2.54
    fig_width_cm = 15  # 宽度，厘米
    fig_height_cm = 12  # 高度，厘米
    # 创建双轴图形
    fig, ax1 = plt.subplots(figsize=(fig_width_cm*cm_to_inch, fig_height_cm*cm_to_inch))
    
    # 第一条曲线 (左轴)
    ax1.set_xlabel('Service Time (s)', fontsize=14)
    ax1.set_ylabel(f'{label1} Sum Rate (bps/Hz)', color=color1, fontsize=14)
    line1 = ax1.plot(T_list1, rate_list1, color=color1, linestyle=style1, linewidth=2,
                     marker=marker1, markersize=6, markeredgewidth=2,
                     label=label1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 第二条曲线 (右轴)
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'{label2} Sum Rate (bps/Hz)', color=color2, fontsize=14)
    line2 = ax2.plot(T_list2, rate_list2, color=color2, linestyle=style2, linewidth=2,
                     marker=marker2, markersize=6, markeredgewidth=2, markerfacecolor='none',
                     label=label2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', fontsize=14)
    
    # 添加网格线 (仅适用于左轴)
    ax1.grid(True, alpha=0.3)

    # 设置框和刻度线 - 类似Matlab的box on
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    
    # 设置刻度线朝内并出现在所有四个边上
    ax1.tick_params(axis='both', which='both', direction='in', 
                    top=True, bottom=True, left=True, right=True)
    ax2.tick_params(axis='y', which='both', direction='in', right=True)
    
    # 还可以设置次刻度线
    from matplotlib.ticker import AutoMinorLocator
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
    # # 优化两个Y轴的范围，使曲线在视觉上大致匹配
    # ax1_range = max(rate_list1) - min(rate_list1)
    # ax2_range = max(rate_list2) - min(rate_list2)
    
    # padding1 = ax1_range * 0.1
    # padding2 = ax2_range * 0.1
    
    # ax1.set_ylim([min(rate_list1) - padding1, max(rate_list1) + padding1])
    # ax2.set_ylim([min(rate_list2) - padding2, max(rate_list2) + padding2])
    ax1.set_ylim(ylim1)
    ax2.set_ylim(ylim2)
    # 创建 fig 目录（如果不存在）
    if not os.path.exists('fig'):
        os.makedirs('fig')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(f'fig/{output_name}_dual_axis.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}_dual_axis.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}_dual_axis.png', dpi=300, bbox_inches='tight')
    
    # 显示图片
    plt.show()
    
    # 输出一些基本统计信息
    print("\n统计信息:")
    print(f"{label1} 平均速率: {np.mean(rate_list1):.4f} bps/Hz")
    print(f"{label2} 平均速率: {np.mean(rate_list2):.4f} bps/Hz")

if __name__ == "__main__":
    compare_rates_dual_axis(
        'data/Whole_Service_S2_U3_N4_M6400_Random0.npy', 'WMMSE',
        'data/Whole_Service_S2_U3_N4_M6400_Random0_MRC1.npy', 'MRT',
        time_interval=15,
        output_name="WMMSE_MRC_Comparison",
        color1='#1d73b6ff',  # 蓝色
        color2='#24a645',  # 绿色
        style1='-', 
        style2='--',
        marker1='o', 
        marker2='o',
        ylim1=[6.6, 12.6],  # 左侧y轴范围
        ylim2=[1.7516, 1.7576]   # 右侧y轴范围
    )