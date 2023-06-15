"""main for FedETuning"""
import argparse
import os
import sys
import itertools as it
from loguru import logger
from configs.tuning import hyperparameter_grid
from utils import registry
from utils import build_config
from utils import setup_logger, setup_imports


def main(args=None):

    setup_imports()
    setup_logger()
    config = build_config(args)
    trainer = registry.get_fl_class(config.federated_config.fl_algorithm)()
    trainer.train()


if __name__ == "__main__":
    main()
