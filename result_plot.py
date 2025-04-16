import numpy as np
import matplotlib.pyplot as plt
import os

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 14

def compare_rates(*args, time_interval=10, max_time=None, output_name="rate_comparison", colors=None, markerfacecolors=None, styles=None, markers=None):
    """
    读取并比较多个速率数据文件
    
    参数:
        *args: 交替的文件路径和标签，如 file1, label1, file2, label2, ...
        time_interval: 时间采样间隔
        max_time: 最大显示时间 (可选)
        output_name: 输出文件名前缀
        colors: 线条颜色列表
        styles: 线条样式列表
    """
    # 处理参数
    files = args[::2]  # 偶数位置是文件路径
    labels = args[1::2]  # 奇数位置是标签
    
    # 默认颜色和样式
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    if styles is None:
        styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    if markers is None:
        markers = ['o', 's', '^', 'D', 'x', '+', '*', 'P']
    if markerfacecolors is None:
        markerfacecolors = colors.copy()
    
    # 确保颜色和样式足够
    while len(colors) < len(files):
        colors.extend(colors)
    while len(styles) < len(files):
        styles.extend(styles)

    cm_to_inch = 1/2.54
    fig_width_cm = 15  # 宽度，厘米
    fig_height_cm = 12  # 高度，厘米
    # 创建图形
    fig, ax = plt.subplots(figsize=(fig_width_cm*cm_to_inch, fig_height_cm*cm_to_inch))
    
    # 加载数据并绘图
    all_rates = []
    max_rate = 0
    min_rate = float('inf')
    
    for i, (file_path, label) in enumerate(zip(files, labels)):
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取数据
        rate_list = np.load(file_path)
        all_rates.append(rate_list)
        
        # 记录最大最小值用于调整y轴
        max_rate = max(max_rate, np.max(rate_list))
        min_rate = min(min_rate, np.min(rate_list))
        
        # 生成时间列表
        T_list = np.arange(0, len(rate_list) * time_interval, time_interval)
        
        # 如果指定了最大时间，进行截断
        if max_time:
            idx = np.where(T_list <= max_time)[0]
            T_list = T_list[idx]
            rate_list = rate_list[idx]
        
        # 绘制曲线
        plt.plot(T_list, rate_list, 
                 color=colors[i], linestyle=styles[i], linewidth=2, 
                 marker=markers[i], markeredgecolor=colors[i], markersize=6, markeredgewidth=2, markerfacecolor=markerfacecolors[i],
                 label=label
                 )
    
    # 添加标签和图例
    plt.xlabel('Service Time (s)', fontsize=14)
    plt.ylabel('Sum Rate (bps/Hz)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower center')
    
    # 设置框和刻度线 - 类似Matlab的box on
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # 设置刻度线朝内并出现在所有四个边上
    ax.tick_params(axis='both', which='both', direction='in', 
               top=True, bottom=True, left=True, right=True)
    
    # 还可以设置次刻度线
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # 优化Y轴范围，留出一点边距
    padding = (max_rate - min_rate) * 0.05
    plt.ylim([min_rate - padding, max_rate + padding])
    
    # 创建 fig 目录（如果不存在）
    if not os.path.exists('fig'):
        os.makedirs('fig')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(f'fig/{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'fig/{output_name}.png', dpi=300, bbox_inches='tight')
    
    # 显示图片
    plt.show()
    
    # 输出一些基本统计信息
    print("\n统计信息:")
    for i, (label, rate) in enumerate(zip(labels, all_rates)):
        print(f"{label} 平均速率: {np.mean(rate):.4f} bps/Hz")
    
    # # 计算性能提升
    # if len(all_rates) > 1:
    #     print("\n性能提升:")
    #     base_rate = np.mean(all_rates[0])
    #     for i, (label, rate) in enumerate(zip(labels[1:], all_rates[1:]), 1):
    #         diff = np.mean(rate) - base_rate
    #         percent = (diff / base_rate) * 100
    #         print(f"{labels[0]} vs {label}: {diff:.4f} bps/Hz ({percent:.2f}%)")

if __name__ == "__main__":
    compare_rates(
        # 'data/Whole_Service_S2_U3_N4_M6400_Random0.npy', 'RIS with optimised-phase elements',
        # 'data/Whole_Service_S2_U3_N4_M6400_Random1.npy', 'RIS with random-phase elements',
        # 'data/Whole_Service_S2_U3_N4_M0_Random0.npy', 'RIS without elements',
        # output_name="RIS_comparison",
        # colors=['#1d73b6', '#24a645', '#f27830'],
        # markerfacecolors=['#1d73b6', '#24a645', '#f27830'],
        # styles=['-', '--', '--'],
        # markers=['o', 'o', '+' ],
        'data/Whole_Service_S2_U3_N4_M6400_Random0.npy', 'RIS-ssisted dual-satellite',
        'data/Whole_Service_S1_U3_N4_M6400_Random0_MRC0_R.npy', 'RIS-ssisted satellite S1',
        'data/Whole_Service_S1_U3_N4_M6400_Random0_MRC0_L.npy', 'RIS-ssisted satellite S2',
        output_name="SAT_comparison",
        colors=['#1d73b6', '#24a645', '#f27830'],
        markerfacecolors=['#1d73b6', 'none', 'none'],
        styles=['-', '--', '--'],
        markers=['o', 'o', 'o' ],
        time_interval=15
    )