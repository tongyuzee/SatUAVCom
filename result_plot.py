import numpy as np
import matplotlib.pyplot as plt
import os

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True

def compare_rates(file1='data/Rate_list1.npy', file2='data/Rate_list2.npy', 
                  label1='Algorithm 1', label2='Algorithm 2', 
                  time_interval=10, max_time=None):
    """
    读取并比较两个速率数据文件
    
    参数:
        file1, file2: 要比较的数据文件路径
        label1, label2: 图例标签
        time_interval: 时间采样间隔
        max_time: 最大显示时间 (可选)
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
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 绘制两条曲线
    plt.plot(T_list1, rate_list1, 'b-', linewidth=2, label=label1)
    plt.plot(T_list2, rate_list2, 'g--', linewidth=2, label=label2)
    
    # 添加标签和图例
    plt.xlabel('Service Time (s)', fontsize=14)
    plt.ylabel('Sum Rate (bps/Hz)', fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # 设置坐标轴刻度字体大小
    plt.xticks()
    plt.yticks()
    
    # 创建 fig 目录（如果不存在）
    if not os.path.exists('fig'):
        os.makedirs('fig')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('fig/rate_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('fig/rate_comparison.svg', format='svg', bbox_inches='tight')
    
    # 显示图片
    plt.show()
    
    # 输出一些基本统计信息
    print(f"{label1} 平均速率: {np.mean(rate_list1):.4f} bps/Hz")
    print(f"{label2} 平均速率: {np.mean(rate_list2):.4f} bps/Hz")
    print(f"速率提升: {(np.mean(rate_list1) - np.mean(rate_list2))/np.mean(rate_list2)*100:.2f}%")

if __name__ == "__main__":
    compare_rates(
        file1='data/Rate_list1.npy', 
        file2='data/Rate_list2.npy',
        label1='Total power constaint', 
        label2='PerSat power constraint',
        time_interval=10
    )