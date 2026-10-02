import argparse
import torch

from cellvit.models.cell_segmentation.cellvit_256 import CellViT256


def load_state(path):
    ck = torch.load(path, map_location="cpu")

    if isinstance(ck, dict) and "model_state_dict" in ck:
        ck = ck["model_state_dict"]

    new_ck = {}
    for k, v in ck.items():
        if k.startswith("module."):
            k = k[7:]
        new_ck[k] = v

    return new_ck


parser = argparse.ArgumentParser()
parser.add_argument("--encoder", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

model = CellViT256(
    model256_path=args.encoder,
    num_nuclei_classes=6,
    num_tissue_classes=19,
)

print("正在加载 encoder...")
model.load_pretrained_encoder(args.encoder)

print("正在加载完整 CellViT 权重...")
state_dict = load_state(args.model)
result = model.load_state_dict(state_dict, strict=True)

print("加载成功")
print(result)