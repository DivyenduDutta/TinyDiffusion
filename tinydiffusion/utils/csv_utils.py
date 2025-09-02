import os
import pandas as pd

from tinydiffusion.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger

BENCHMARK_SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "benchmarks"
)
os.makedirs(BENCHMARK_SAVE_PATH, exist_ok=True)


def save_results_to_csv(results: list) -> None:
    """
    Save benchmark results to a CSV file.

    Args:
        results (list): A list of benchmark results to save.
    """
    df = pd.DataFrame(results)
    csv_path = os.path.join(BENCHMARK_SAVE_PATH, "benchmark_results.csv")

    file_exists = os.path.isfile(csv_path)

    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
    LOGGER.info(f"Saved benchmark results to {csv_path}")


def load_results_from_csv() -> pd.DataFrame:
    """
    Load benchmark results from a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark results.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = os.path.join(BENCHMARK_SAVE_PATH, "benchmark_results.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        LOGGER.info(f"Loaded benchmark results from {csv_path}")
        return df
    else:
        LOGGER.warning(f"No benchmark results found at {csv_path}")
        raise FileNotFoundError(f"Benchmark results CSV not found at {csv_path}.")
