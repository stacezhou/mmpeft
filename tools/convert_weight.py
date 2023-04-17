from mmpretrain.models.classifiers.clip import load
import torch
def parse_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    return (
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='ViT-B/32')
parser.add_argument('--download_root')
parser.add_argument('--save_path')
args = parser.parse_args()
clip, prepocess = load(args.arch, download_root=args.download_root)
torch.save(clip.state_dict(), args.save_path)
args = parse_model(clip.state_dict())
config = f"""
embed_dim = {args[0]}, 
image_resolution = {args[1]},
vision_layers = {args[2]},
vision_width = {args[3]},
vision_patch_size = {args[4]},
context_length = {args[5]},
vocab_size = {args[6]},
transformer_width = {args[7]},
transformer_heads = {args[8]},
transformer_layers = {args[9]},
"""