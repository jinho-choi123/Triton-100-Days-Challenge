import pytest
from pathlib import Path
from loguru import logger
import shutil


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def bench_result_dir(project_root: Path) -> Path:
    result_dir = project_root / "bench_results"
    if result_dir.exists():
        logger.info(
            "Removing existing benchmark results directory... All the previous benchmark results will be removed."
        )
        shutil.rmtree(result_dir)
    result_dir.mkdir()
    logger.info(f"Benchmark results directory {result_dir} created.")
    return result_dir
