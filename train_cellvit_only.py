# -*- coding: utf-8 -*-

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

import wandb

from cellvit.training.base_ml.base_cli import ExperimentBaseParser
from cellvit.training.experiments.experiment_cellvit_pannuke import (
    ExperimentCellVitPanNuke,
)


def main():
    parser = ExperimentBaseParser()
    configuration = parser.parse_arguments()

    dataset_name = configuration["data"]["dataset"].lower()

    if dataset_name != "pannuke":
        raise ValueError(
            "该入口要求 data.dataset 为 PanNuke，"
            f"当前为 {configuration['data']['dataset']}"
        )

    experiment = ExperimentCellVitPanNuke(
        default_conf=configuration
    )

    outdir = experiment.run_experiment()

    print("\n训练结束")
    print("run directory:", outdir)
    print(
        "best checkpoint:",
        os.path.join(outdir, "checkpoints", "model_best.pth"),
    )

    wandb.finish()


if __name__ == "__main__":
    main()