from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

SRC = ROOT / "src"

sys.path.insert(0, str(SRC))
from fedaegis.config import Config
from fedaegis.data.iris_loader import IrisLoader


cfg = Config()

from fedaegis.data.dataset_factory import DatasetFactory

loader = DatasetFactory.create(cfg)

X_train, X_test, y_train, y_test = loader.load()

print("=" * 45)
print("FedAegis-X")
print("=" * 45)

print("Dataset Loaded")

print(f"Train Samples : {len(X_train)}")

print(f"Test Samples  : {len(X_test)}")

print(f"Features      : {X_train.shape[1]}")

print(f"Classes       : {len(set(y_train))}")

from fedaegis.client import Client
from fedaegis.models.logistic import LogisticModel
from fedaegis.metrics.classification import evaluate

from fedaegis.data.partition import iid_partition

from fedaegis.data.noniid import DirichletPartitioner

partitioner = DirichletPartitioner(

    alpha=cfg["federation"]["partition"]["alpha"]

)

parts = partitioner.split(

    X_train,

    y_train,

    cfg["federation"]["clients"]

)

clients = []

from fedaegis.models.model_factory import ModelFactory


for idx, (x, y) in enumerate(parts):

    model = ModelFactory.create(

    cfg["model"]["type"]

)

    client = Client(
        idx,
        model,
        x,
        y
    )

    clients.append(client)

updates = []

for c in clients:

    updates.append(
        c.train()
    )

from fedaegis.aggregation.fedavg import FedAvgAggregator
from fedaegis.core.server import Server
from fedaegis.core.history import History
from fedaegis.core.trainer import FederatedTrainer
from fedaegis.outputs.writer import ResultWriter

from fedaegis.core.aggregation_factory import AggregationFactory

aggregator = AggregationFactory.create(

    cfg["federation"]["aggregation"]["type"]

)

server = Server(
    aggregator
)

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

writer = ResultWriter(

    cfg["output"]["directory"]

)

writer.save_metrics(
    history
)

history.save_json(

    "outputs/history.json"

)

print()

print("=" * 40)

print("Training Finished")

print("=" * 40)
