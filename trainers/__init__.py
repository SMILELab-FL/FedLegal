"""FedLegal's trainers registry in trainer.__init__.py -- IMPORTANT!"""

from trainers.FedBaseTrainer import BaseTrainer
from run.fedavg.trainer import FedAvgTrainer
from run.centralized.trainer import CenClientTrainer
from run.dry_run.trainer import DryTrainer
from run.fedprox.trainer import FedProxTrainer
from run.ditto.trainer import DittoTrainer
from run.fedopt.trainer import FedOptTrainer
from run.local.trainer import LocalTrainer


__all__ = [
    "BaseTrainer",
    "FedAvgTrainer",
    "CenClientTrainer",
    "DryTrainer",
    "FedProxTrainer",
    "DittoTrainer",
    "FedOptTrainer",
    "LocalTrainer",
]
