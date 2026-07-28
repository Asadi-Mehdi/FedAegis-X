from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

SRC = ROOT / "src"

sys.path.insert(0, str(SRC))
from fedaegis.config import Config
from fedaegis.data.iris_loader import IrisLoader


cfg = Config()

loader = IrisLoader(
    test_size=cfg["dataset"]["test_size"],
    random_state=cfg["dataset"]["random_state"]
)

X_train, X_test, y_train, y_test = loader.load()

print("=" * 45)
print("FedAegis-X")
print("=" * 45)

print("Dataset Loaded")

print(f"Train Samples : {len(X_train)}")

print(f"Test Samples  : {len(X_test)}")

print(f"Features      : {X_train.shape[1]}")

print(f"Classes       : {len(set(y_train))}")
