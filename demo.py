import torch
import os
from PIL import Image

from xai_dataset import XAIDataset, get_img_pair_from_paths
from example_loaders import toy_df, wildme_multispecies_miewid
from core import explain

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, img_size, img_transforms = wildme_multispecies_miewid(device)

    df = toy_df()
    dataset = XAIDataset(df, img_size, img_transforms)

    img_0, img_1, img_np_0, img_np_1 = dataset.get_img_pair(device, "cow_0_0", "cow_0_1")
    #img_0, img_1, img_np_0, img_np_1 = get_img_pair_from_paths(device, "data/cow_0_1.jpg", "data/cow_0_0.jpg", img_size, img_transforms)

    img = explain(device,
                    img_0,
                    img_1,              # first annot
                    img_np_0,              # second annot
                    img_np_1,
                    model,
                    ["backbone.blocks.3"],  # intermediate layer to visualize
                    k_lines=20,             # number of matches to visualize as lines
                    k_colors=10,            # number of matches to visualize as colors
                    )[0]
    
    img = Image.fromarray(img)
    img.save("examples/cow_pairx_example.png")

if __name__ == "__main__":
    main()


