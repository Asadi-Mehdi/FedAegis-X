from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))


from fedaegis.config import Config

from fedaegis.data.iris_loader import IrisLoader
from fedaegis.data.partition import iid_partition

from fedaegis.models.logistic import LogisticModel

from fedaegis.client import Client

from fedaegis.core.server import Server
from fedaegis.core.history import History
from fedaegis.core.trainer import FederatedTrainer

from fedaegis.core.aggregation_factory import (
    AggregationFactory
)

from fedaegis.outputs.writer import ResultWriter

from fedaegis.cost.matrix import CostMatrix
from fedaegis.cost.metrics import CostMetrics

from fedaegis.evaluation.report import EvaluationReport


# ============================================================
# Configuration
# ============================================================

cfg = Config()


# ============================================================
# Dataset
# ============================================================

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


# ============================================================
# IID Client Partitioning
# ============================================================

parts = iid_partition(
    X_train,
    y_train,
    cfg["federated"]["clients"]
)


# ============================================================
# Clients
# ============================================================

clients = []

for client_id, (X_client, y_client) in enumerate(parts):

    model = LogisticModel()

    client = Client(
        client_id=client_id,
        model=model,
        X=X_client,
        y=y_client
    )

    clients.append(client)


# ============================================================
# Aggregator
# ============================================================

aggregation_name = (
    cfg["federated"]
    .get("aggregation", {})
    .get("type", "fedavg")
)

aggregator = AggregationFactory.create(
    aggregation_name
)


server = Server(
    aggregator=aggregator
)


# ============================================================
# Federated Training
# ============================================================

history = History()

trainer = FederatedTrainer(
    clients=clients,
    server=server,
    history=history,
    rounds=cfg["federated"]["rounds"]
)

global_model = trainer.fit(
    X_test,
    y_test
)


# ============================================================
# Results
# ============================================================

output_directory = (
    cfg["output"]["directory"]
)

writer = ResultWriter(
    output_directory
)

writer.save_metrics(
    history
)

history.save_json(
    f"{output_directory}/history.json"
)


# ============================================================
# Cost Evaluation
# ============================================================

cost_matrix = CostMatrix(
    false_positive=1.0,
    false_negative=10.0
)

prediction = global_model.predict(
    X_test
)

total_cost = CostMetrics.total_cost(
    y_test,
    prediction,
    cost_matrix
)

fnr = CostMetrics.false_negative_rate(
    y_test,
    prediction
)


report = EvaluationReport()

report.add(
    "total_cost",
    float(total_cost)
)

report.add(
    "false_negative_rate",
    float(fnr)
)

report.save(
    f"{output_directory}/cost_report.json"
)


# ============================================================
# Final Output
# ============================================================

print()

print("=" * 45)
print("Cost Evaluation")
print("=" * 45)

print(
    f"Total Cost : {total_cost:.4f}"
)

print(
    f"FNR        : {fnr:.4f}"
)

print()

print("=" * 45)
print("Training Finished")
print("=" * 45)
