from pathlib import Path

import torch
import torch_pruning as tp
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

from inference import get_inference_device


def prune_efficientnet_b7(
    output_path: Path = Path("efficientnet_b7_pruned.pth"),
    pruning_ratio: float = 0.3,
) -> None:
    device = get_inference_device()
    model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT).to(device).eval()

    example_inputs = torch.randn(1, 3, 600, 600, device=device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    importance = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = [model.classifier]

    pruner = tp.pruner.MagnitudePruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=8
    )
    pruner.step()

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print()
    print(f"Parameters:  {base_params / 1e6:.2f} M -> {pruned_params / 1e6:.2f} M")
    print(f"MACs:        {base_macs / 1e9:.2f} GMACs -> {pruned_macs / 1e9:.2f} GMACs")
    print()

    torch.save(model, output_path)
    print(f"Saved pruned model to: {output_path}")
    print()


if __name__ == "__main__":
    prune_efficientnet_b7()