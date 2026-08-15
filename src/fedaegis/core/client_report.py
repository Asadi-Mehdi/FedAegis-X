from dataclasses import dataclass


@dataclass
class ClientReport:

    client_id: int

    samples: int

    fnr: float

    class_balance: float

    reliability: float

    parameters: dict
