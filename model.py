# model.py
# (Existing code from the original model.py remains, and modifications
# are marked with comments like this: # MODIFIED: ...)
# ==============================================================================
#   MODIFIED: Integration of Pattention and nGPT Normalization
# ==============================================================================
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    # Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # Apply the rotary embeddings to the query and key
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)
    return q_rot, k_rot

def get_sinusoidal_embeddings( n_positions, dim):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim))
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb


class Pattention(nn.Module):
    """Pattention Layer from Tokenformer, modified for nGPT."""

    def __init__(
        self,
        config, # MODIFIED: use config instead of neox_args
        input_channels,
        output_channels,
        param_token_num,
        param_key_init_method,
        param_value_init_method,
    ):
        super().__init__()

        self.config = config # MODIFIED: store config
        self.param_token_num = param_token_num
        self.param_key_dim = input_channels
        self.param_value_dim = output_channels
        self.norm_activation_type = 'gelu_l2_norm' # MODIFIED: Hardcode gelu_l2_norm for now, can be set in config later
        
        self.key_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_key_dim), dtype=torch.bfloat16))  # MODIFIED: bfloat16
        self.value_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_value_dim), dtype=torch.bfloat16))  # MODIFIED: bfloat16
        
        param_key_init_method(self.key_param_tokens)
        param_value_init_method(self.value_param_tokens)


    def nonlinear_norm_func(self, inputs, normalize_type, dim=-1):
        if normalize_type == 'softmax': 
            # NOTE: softmax = exp_l1_norm
            # outputs = F.softmax(inputs, dim=dim) * inputs.shape[dim]
            nonlinear_outputs = torch.exp(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=1, dim=dim, keepdim=True) * inputs.shape[dim]
            outputs = norm_outputs
        elif normalize_type == 'gelu_l2_norm':
            nonlinear_outputs = F.gelu(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True) * math.sqrt(nonlinear_outputs.shape[dim])
            outputs = norm_outputs
        elif normalize_type == 'l2_norm_gelu':
            norm_outputs = inputs / torch.norm(inputs, p=2, dim=dim, keepdim=True) * math.sqrt(inputs.shape[dim])
            nonlinear_outputs = F.gelu(norm_outputs)
            outputs = nonlinear_outputs
        else:
            raise NotImplementedError
        return outputs

    def forward(self, inputs, dropout_p=0.0, router_index=None, attn_mask=None, scale=None):

        query = inputs
        if router_index is None:
            # not MoE mode
            key, value = self.key_param_tokens, self.value_param_tokens
        else:
            key, value = self.key_param_tokens[router_index], self.value_param_tokens[router_index]
        
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 if scale is None else scale 
        # just for gelu nonlinear, set torch.zeros for softmax
        attn_bias = torch.ones(L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # just for gelu nonlinear, set -inf for softmax
                attn_bias.masked_fill_(attn_mask.logical_not(), 0)
            else:
                raise NotImplementedError

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        # just for gelu nonlinear, set attn_weight += attn_bias for softmax
        attn_weight *= attn_bias
        # modified softmax
        attn_weight = self.nonlinear_norm_func(attn_weight, self.norm_activation_type, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = attn_weight @ value

        return output

class Block(nn.Module):

    def __init__(self, config, iblock):
        super().__init__()
        self.config = config
        # MODIFIED: Replace linear layers with Pattention
        self.key = Pattention(
            config=config,
            input_channels=config.n_embd,
            output_channels=config.n_embd,
            param_token_num=config.qkv_slot_num, # MODIFIED: use config variable
            param_key_init_method=nn.init.xavier_normal_,
            param_value_init_method=nn.init.zeros_
        )

        self.query = Pattention(
            config=config,
            input_channels=config.n_embd,
            output_channels=config.n_embd,
            param_token_num=config.qkv_slot_num, # MODIFIED: use config variable
            param_key_init_method=nn.init.xavier_normal_,
            param_value_init_method=nn.init.zeros_
        )
        self.value = Pattention(
            config=config,
            input_channels=config.n_embd,
            output_channels=config.n_embd,
             param_token_num=config.qkv_slot_num, # MODIFIED: use config variable
            param_key_init_method=nn.init.xavier_normal_,
            param_value_init_method=nn.init.zeros_
        )

        self.att_c_proj = Pattention(
            config=config,
            input_channels=config.n_embd,
            output_channels=config.n_embd,
             param_token_num=config.proj_slot_num,  # MODIFIED: use config variable
            param_key_init_method=nn.init.xavier_normal_,
            param_value_init_method=nn.init.zeros_
        )

        self.c_fc    = Pattention(
            config=config,
            input_channels=config.n_embd,
            output_channels=2 * 4 * config.n_embd,
             param_token_num=config.ffn_slot_num, # MODIFIED: use config variable
            param_key_init_method=nn.init.xavier_normal_,
            param_value_init_method=nn.init.zeros_
        )
        self.silu    = nn.SiLU()
        self.mlp_c_proj  = Pattention(
            config=config,
            input_channels=4 * config.n_embd, # MODIFIED: Use 4*n_embd as input
            output_channels=config.n_embd,
             param_token_num=config.ffn_slot_num,  # MODIFIED: use config variable
            param_key_init_method=nn.init.xavier_normal_,
            param_value_init_method=nn.init.zeros_
        )

        if (config.use_nGPT == 0):
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)
        
        # MODIFIED: Initialize nGPT parameters if use_nGPT=1
        if (config.use_nGPT == 1):
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.sqk_init_value = 1.0      
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32)) # MODIFIED: Keep it as n_embd

            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(self.suv_init_scaling*torch.ones(2 * 4 * config.n_embd, dtype=torch.float32))# MODIFIED: Change it to 2 * 4 * n_embd

    def justnorm(self, x):
        # MODIFIED: implementation of the hypersphere normalization
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, h):
        B, T, C = h.size()

        hin = h
        if (self.config.use_nGPT == 0):
            hin = self.rmsnorm_att(h)

        q = self.query(hin).view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        k = self.key(hin).view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        v = self.value(hin).view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        
        sinusoidal_pos = get_sinusoidal_embeddings(T, self.config.n_embd // self.config.n_head).to(device=q.device)
        q, k = apply_rotary_position_embeddings(sinusoidal_pos, q.transpose(1, 2), k.transpose(1, 2))
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)

        if (self.config.use_nGPT == 1):
            sqk = (self.sqk * (self.sqk_init_value/self.sqk_init_scaling)).view(1, 1, self.config.n_head, self.config.n_embd // self.config.n_head)
            q = sqk * self.justnorm(q) 
            k = sqk * self.justnorm(k)

        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5
        if (self.config.use_nGPT == 0): softmax_scale = 1.0 / sqrt_head_dim 
        if (self.config.use_nGPT == 1): softmax_scale = sqrt_head_dim
        y = flash_attn_func(q.to(dtype=torch.bfloat16), k.to(dtype=torch.bfloat16), v.to(dtype=torch.bfloat16), dropout_p=0.0, softmax_scale=softmax_scale, causal=True, window_size=(-1, -1), alibi_slopes=None, deterministic=True)
        y = y.to(dtype=q.dtype)
        y = y.contiguous().view(B, T, self.config.n_embd)
        
        h_att = self.att_c_proj(y)
        if (self.config.use_nGPT == 0):
            h = h + h_att
        if (self.config.use_nGPT == 1):
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)
            
            A_norm = self.justnorm(h) # normally, normalization is not needed
            B_norm = self.justnorm(h_att)
                
            #res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        hin = h
        if (self.config.use_nGPT == 0):
            hin = self.rmsnorm_mlp(h)
        uv = self.c_fc(hin)
        if (self.config.use_nGPT == 1):
            suv = (self.suv * ((self.suv_init_value/self.suv_init_scaling) * (self.config.n_embd ** 0.5))) 
            uv = suv * uv   
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)
        
        if (self.config.use_nGPT == 0):
            h = h + h_mlp
        if (self.config.use_nGPT == 1):
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = self.justnorm(h) # normally, normalization is not needed
            B_norm = self.justnorm(h_mlp)
                
            #res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)
        return h

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0 ** 0.5)    # 1 / sqrt(n_embd)
    use_nGPT: int = 0 # MODIFIED: Add nGPT configuration
    dropout: float = 0.0
    bias: bool = False
    qkv_slot_num: int = 2140 # MODIFIED: Added new parameters for Pattention
    ffn_slot_num: int = 8560 # MODIFIED: Added new parameters for Pattention
    proj_slot_num: int = 2140 # MODIFIED: Added new parameters for Pattention


class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, il) for il in range(config.n_layer)])
        ))
        # MODIFIED: Use Pattention for the lm_head
        self.lm_head = Pattention(
            config=config,
            input_channels=config.n_embd,
            output_channels=config.vocab_size,
            param_token_num=config.vocab_size, # MODIFIED: use vocab size for parameter tokens
            param_key_init_method=nn.init.xavier_normal_,
            param_value_init_method=nn.init.zeros_
        )
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # *we don't use it becuase in the nGPT paper there was no weight tying of weights*
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale/math.sqrt(2 * config.n_layer))
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
        if (config.use_nGPT == 1):
            self.sz_init_value = 1.00
            self.sz_init_scaling = config.base_scale
            self.sz = torch.nn.Parameter(self.sz_init_scaling*torch.ones(config.vocab_size, dtype=torch.float32))

        if (config.use_nGPT == 0):
            self.rmsnorm_f = RMSNorm(config.n_embd)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        #assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        x = tok_emb
        for block in self.transformer.h:
            x = block(x)

        if (self.config.use_nGPT == 0):
            x = self.rmsnorm_f(x)

        # MODIFIED: Use Pattention lm_head with scaled logits
        logits = self.lm_head(x)
        if (self.config.use_nGPT == 1):
            sz = self.sz * (self.sz_init_value/self.sz_init_scaling)
            logits = sz * logits

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = logits[:, [-1], :] # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = False#fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer