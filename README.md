Simple implementation of low-rank adaptation of large language models (LoRA) for [Segment Anything 2](https://github.com/facebookresearch/sam2). LoRA is only applied to image encoder attention blocks, similar to [this repository](https://github.com/JamesQFreeman/Sam_LoRA). 

The code assumes standard preprocessing for SAM (e.g. normalization) has already been performed and uses no prompts by default, but can be customized easily. This repository is still in development so please report any issues!

### Important Information (please read):

In my own testing, LoRA didn't seem to benefit SAM 2 as much as it did SAM 1. The accuracy, number of epochs for convergence, and memory consumption remained more or less the same compared to training regular SAM 2. The training duration reduced slightly, but the inference duration also increased by a little. Either LoRA could be applied better to this algorithm than currently in this code, or it isn't as effective for Hiera compared to the original SAM 1 Vision Transformer. 
