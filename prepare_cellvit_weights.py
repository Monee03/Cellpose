from pathlib import Path
import argparse
import torch


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError("checkpoint 不是字典")

    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        new_state_dict[key] = value

    return new_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(input_path, map_location="cpu")
    state_dict = get_state_dict(checkpoint)

    full_state_path = output_dir / (
        input_path.stem + "_state_dict.pth"
    )
    torch.save(state_dict, full_state_path)

    # 提取 encoder 权重
    encoder_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("encoder."):
            new_key = key[len("encoder."):]
            encoder_state_dict[new_key] = value

    encoder_path = output_dir / (
        input_path.stem + "_encoder.pth"
    )
    torch.save(encoder_state_dict, encoder_path)

    print("原始权重:", input_path)
    print("模型 state_dict:", full_state_path)
    print("encoder state_dict:", encoder_path)
    print("模型参数张量数:", len(state_dict))
    print("encoder 参数张量数:", len(encoder_state_dict))


if __name__ == "__main__":
    main()