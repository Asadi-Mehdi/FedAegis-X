from pathlib import Path
import logging


def build_logger(log_dir):

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=Path(log_dir) / "training.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s"
    )

    return logging.getLogger("FedAegis")
