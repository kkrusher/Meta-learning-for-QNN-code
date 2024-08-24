import logging
import matplotlib.pyplot as plt
from pathlib import Path
import json5
import numpy as np


def plot_losses(losses_file, horizontal_line=None, fig_path=None, show_plot=False):
    """
    加载并绘制训练损失曲线，并可选择性地添加水平虚线。

    参数:
    - losses_file (str 或 Path): 实验保存结果的路径。
    - horizontal_line (float 或 None): 如果不为 None，在图中绘制该值的水平虚线。

    返回:
    - None
    """
    # 确保 losses_file 是一个 Path 对象
    losses_file = Path(losses_file)

    # 检查文件是否存在
    if not losses_file.exists():
        logging.info(f"Error: '{losses_file}' 文件不存在。")
        return

    # 加载 losses
    with open(losses_file, "r") as f:
        losses = json5.load(f)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", label="Loss")

    # 如果提供了水平线的值，绘制水平虚线
    if horizontal_line is not None:
        plt.axhline(
            y=horizontal_line,
            color="r",
            linestyle="--",
            label=f"Generated loss: {horizontal_line}",
        )

    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    if fig_path is not None:
        plt.savefig(fig_path)
    if show_plot:
        plt.show()
    # 关闭 plt
    plt.close()


# Function to plot loss with error bars and mean final loss line
def plot_loss_with_error_bars(losses_matrix, final_loss_avg, fig_path, show_plot=False):
    losses_mean = np.mean(losses_matrix, axis=0)
    losses_std = np.std(losses_matrix, axis=0)

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        range(1, len(losses_mean) + 1),
        losses_mean,
        yerr=losses_std,
        fmt="-o",
        capsize=5,
        label="Mean Loss with Error Bars",
    )
    plt.axhline(
        y=final_loss_avg,
        color="r",
        linestyle="--",
        label=f"Mean Final Loss: {final_loss_avg:.4f}",
    )

    plt.title(f"Training Loss with Error Bars")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if fig_path is not None:
        plt.savefig(fig_path)
    if show_plot:
        plt.show()
    plt.close()
