import os
import pandas as pd
import matplotlib.pyplot as plt

from tinydiffusion.utils.csv_utils import load_results_from_csv, BENCHMARK_SAVE_PATH
from tinydiffusion.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger


def visualize_benchmark_results(df: pd.DataFrame) -> None:
    """
    Visualize benchmark results from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing benchmark results.
    """
    # Set figure with 3 subplots (stacked vertically)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    #  Inference time
    axes[0].bar(
        df["desc"], df["avg_inference_time"], yerr=df["std_inference_time"], capsize=5
    )
    axes[0].set_ylabel("Inference Time (s)")
    axes[0].set_title("Benchmark Comparison")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # CPU memory usage
    axes[1].bar(
        df["desc"],
        df["avg_cpu_mem_usage"],
        yerr=df["std_cpu_mem_usage"],
        capsize=5,
        color="orange",
    )
    axes[1].set_ylabel("CPU Memory (MB)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    # GPU memory usage
    axes[2].bar(
        df["desc"],
        df["avg_gpu_mem_usage"],
        yerr=df["std_gpu_mem_usage"],
        capsize=5,
        color="green",
    )
    axes[2].set_ylabel("GPU Memory (MB)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.7)

    plt.xticks(rotation=15)
    plt.tight_layout()
    viz_save_path = os.path.join(BENCHMARK_SAVE_PATH, "benchmark_comparison.png")
    plt.savefig(viz_save_path)
    LOGGER.info(f"Saved benchmark visualization to {viz_save_path}")


if __name__ == "__main__":
    df = load_results_from_csv()

    visualize_benchmark_results(df)
