from pathlib import Path
import copy
import yaml


PROJECT_ROOT = Path(".").resolve()

TEMPLATE = (
    PROJECT_ROOT
    / "logs/PanNuke/CellViTHV/ViT256/Best-Setting/Fold-1/config.yaml"
)

WEIGHT_ROOT = PROJECT_ROOT / "models/pretrained/converted"
DATA_ROOT = PROJECT_ROOT / "dataset_cellvitpp"


def set_config(dataset_name, magnification, suffix):
    with open(TEMPLATE, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise RuntimeError("官方模板配置为空")

    dataset_path = (
        DATA_ROOT / f"{dataset_name}_pannuke"
    ).resolve()

    if magnification == 40:
        weight_prefix = "CellViT-256-x40"
    else:
        weight_prefix = "CellViT-256-x20"

    encoder_path = (
        WEIGHT_ROOT / f"{weight_prefix}_encoder.pth"
    ).resolve()

    model_path = (
        WEIGHT_ROOT / f"{weight_prefix}_state_dict.pth"
    ).resolve()

    # 基本运行配置
    config["random_seed"] = 19
    config["gpu"] = 0
    config["run_sweep"] = False
    config["eval_checkpoint"] = "model_best.pth"

    # 数据配置
    config.setdefault("data", {})
    config["data"]["dataset"] = "PanNuke"
    config["data"]["dataset_path"] = str(dataset_path)
    config["data"]["train_folds"] = [0]
    config["data"]["val_folds"] = [1]
    config["data"]["test_folds"] = [2]
    config["data"]["input_shape"] = 256
    config["data"]["num_nuclei_classes"] = 6
    config["data"]["num_tissue_classes"] = 19
    config["data"]["magnification"] = magnification

    # ViT256 模型配置
    config.setdefault("model", {})
    config["model"]["backbone"] = "ViT256"
    config["model"]["pretrained_encoder"] = str(encoder_path)
    config["model"]["pretrained"] = str(model_path)
    config["model"]["regression_loss"] = False

    # 训练配置
    config.setdefault("training", {})
    config["training"]["batch_size"] = 4
    config["training"]["epochs"] = 100
    config["training"]["unfreeze_epoch"] = 0
    config["training"]["optimizer"] = "AdamW"
    config["training"]["mixed_precision"] = True
    config["training"]["eval_every"] = 1
    config["training"]["early_stopping_patience"] = 15

    config["training"].setdefault(
        "optimizer_hyperparameter",
        {
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "betas": [0.9, 0.999],
        },
    )

    config["training"].setdefault(
        "scheduler",
        {"scheduler_type": "exponential"},
    )

    # 日志配置
    config.setdefault("logging", {})
    config["logging"]["mode"] = "offline"
    config["logging"]["project"] = "cellvit_baseline"
    config["logging"]["notes"] = (
        f"CellViT256 fine-tuning on {dataset_name}"
    )
    config["logging"]["log_comment"] = (
        f"cellvit256_{dataset_name}"
    )
    config["logging"]["log_dir"] = str(
        PROJECT_ROOT / f"logs_local/{dataset_name}"
    )
    config["logging"]["wandb_dir"] = str(
        PROJECT_ROOT / f"logs_local/{dataset_name}/wandb"
    )

    out_path = PROJECT_ROOT / f"config_{dataset_name}.yaml"

    with open(out_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print("生成:", out_path)
    print("dataset_path:", dataset_path)
    print("pretrained_encoder:", encoder_path)
    print("pretrained:", model_path)


if __name__ == "__main__":
    set_config("consep", 40, "consep")
    set_config("lizard", 20, "lizard")