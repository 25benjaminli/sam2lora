Simple implementation of low-rank adaptation of large language models (LoRA) for [Segment Anything 2](https://github.com/facebookresearch/sam2). LoRA is applied to image encoder attention blocks. 

Inspiration comes from [this repository](https://github.com/JamesQFreeman/Sam_LoRA)

You should be able to use this implementation out of the box with SAM 2, although the code is in development so please report any issues!

**Disclaimer**: in my own testing, the training duration was reduced minimally when testing the algorithm, and inference duration was increased slightly. The accuracy, number of epochs for convergence, and memory consumption remained more or less the same compared to training regular SAM 2. Either LoRA could be applied better to this algorithm than it is currently in this code, or it isn't as effective for Hiera compared to the original SAM 1 Vision Transformer. 
