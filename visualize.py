import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path


def plot_all_equities_advanced(folder_path):
    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 8))

    # 获取所有 CSV
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"错误: 文件夹 {folder_path} 中没有找到 CSV 文件。")
        return

    all_data = []
    print(f"正在处理 {len(csv_files)} 个日志文件...")

    for file in csv_files:
        try:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()

            if 'equity' in df.columns and 'step' in df.columns:
                # 【核心修复 1】：强制转换为数字类型，非数字变为 NaN
                df['equity'] = pd.to_numeric(df['equity'], errors='coerce')
                df['step'] = pd.to_numeric(df['step'], errors='coerce')

                # 去除包含 NaN 的行
                df = df.dropna(subset=['equity', 'step'])

                # 【核心修复 2】：处理重复 step 并确保它是 numeric
                series = df.groupby('step')['equity'].mean()

                # 绘制淡色的单条曲线
                plt.plot(series.index, series.values, color='gray', alpha=0.1, linewidth=0.5)

                all_data.append(series)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")

    if not all_data:
        print("没有可用的有效权益数据。")
        return

    # 合并数据
    combined_df = pd.concat(all_data, axis=1)

    # 【核心修复 3】：确保整个合并后的 DataFrame 都是数字类型，然后再插值
    combined_df = combined_df.apply(pd.to_numeric, errors='coerce')

    # 进行插值和填充
    combined_df = combined_df.interpolate(method='linear').ffill().bfill()

    avg_equity = combined_df.mean(axis=1)
    std_equity = combined_df.std(axis=1)

    # 绘制
    plt.plot(avg_equity.index, avg_equity.values, color='#1f77b4', linewidth=3, label='Average Performance')
    plt.fill_between(avg_equity.index,
                     avg_equity - std_equity,
                     avg_equity + std_equity,
                     color='#1f77b4', alpha=0.2, label='1 Std Dev (Volatility)')

    plt.axhline(y=5000, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
    plt.title(f'BTC PPO Training Analysis: {len(csv_files)} Parallel Episodes', fontsize=16)
    plt.xlabel('Training Steps')
    plt.ylabel('Equity (USDT)')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('equity_summary_report.png')
    print(f"✅ 成功生成报告: equity_summary_report.png")


# 调用
path = r'C:\Users\Chin Jie\Downloads\btc_ppo_smc_bot\logs'
plot_all_equities_advanced(path)