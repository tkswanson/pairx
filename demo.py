import torch
import os
from PIL import Image

from xai_dataset import XAIDataset
from example_loaders import toy_df, wildme_multispecies_miewid
from core import explain

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, img_size, img_transforms = wildme_multispecies_miewid(device)

    df = toy_df()
    dataset = XAIDataset(df, img_size, img_transforms)

    _, img = explain(device,
                     dataset,
                     "cow_0_0",              # first annot
                     "cow_0_1",              # second annot
                     model,
                     ["backbone.blocks.3"],  # intermediate layer to visualize
                     k_lines=20,             # number of matches to visualize as lines
                     k_colors=10,            # number of matches to visualize as colors
                     return_img=True
                     )
    
    img = Image.fromarray(img)
    img.save("examples/cow_pairx_example.png")

if __name__ == "__main__":
    main()


