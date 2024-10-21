import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# import torch parameter
from torch.nn.parameter import Parameter


# base code: https://github.com/JamesQFreeman/Sam_LoRA
class LoRA_SAM2(nn.Module):

    def __init__(self, predictor, r: int, lora_layer=None):
        super(LoRA_SAM2, self).__init__()


        assert r > 0
        self.predictor = predictor

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(self.predictor.model.image_encoder.trunk.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        st = "Total number of parameters sam2 before lora:"
        print(st, sum(p.numel() for p in predictor.model.parameters() if p.requires_grad))

        # lets freeze first
        for param in self.predictor.model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        print(len(self.predictor.model.image_encoder.trunk.blocks))
        for t_layer_i, blk in enumerate(self.predictor.model.image_encoder.trunk.blocks): # TRUNK
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False, device='cuda')
            w_b_linear_q = nn.Linear(r, self.dim, bias=False, device='cuda')
            w_a_linear_v = nn.Linear(self.dim, r, bias=False, device='cuda')
            w_b_linear_v = nn.Linear(r, self.dim, bias=False, device='cuda')
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            self.predictor.model.image_encoder.trunk.blocks[t_layer_i].attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()

    def save_lora_parameters(self, filename: str) -> None:
        # scheduler, optimizer, epoch
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        # if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
        #     state_dict = self.sam.module.state_dict()
        # else:
        #     state_dict = self.sam.state_dict()

        state_dict = self.predictor.model.state_dict()
        for key, value in state_dict.items():
            # print("keys", key)
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors} # mask decoder is being updated
        
        # di = {
        #     "state_dict": merged_dict,
        #     "scheduler": scheduler,
        #     "epoch": epoch,
        #     "optimizer": optimizer,
        # }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.predictor.model.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.predictor.model.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


    def forward(self, image, using_sigmoid=True, return_img_embedding=False, upscale=True):
        self.predictor.set_image_batch(image) # normalized regularly? Need to remove this
        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=None,boxes=None,masks=None)
        high_res_features = [feat_level for feat_level in self.predictor._features["high_res_feats"]]
        # print(self.predictor._features["image_embed"].shape)
        low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
            image_embeddings=self.predictor._features["image_embed"],
            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=True, # !! ??
            high_res_features=high_res_features,
            using_sigmoid=using_sigmoid
        )
        # print(low_res_masks.shape)

        # ! resolve orig_hw thing
        if upscale:
            prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])

        if return_img_embedding:
            return low_res_masks, prd_masks, prd_scores, self.predictor._features["image_embed"]
        
        if upscale:
            return low_res_masks, prd_masks, prd_scores
        else:
            return low_res_masks, prd_scores
      


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv