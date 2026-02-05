from transformers import PretrainedConfig
import torch

class GLMConfig(PretrainedConfig):
    model_type = "glm"
    def __init__(
        self,
        num_layers=36,
        padded_vocab_size=128,
        hidden_size=2560,
        ffn_hidden_size=6832,
        kv_channels=64,
        num_attention_heads=40,
        seq_length=2048,
        hidden_dropout=0.0,
        classifier_dropout=None,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        glu_activation='geglu',
        torch_dtype=torch.bfloat16,
        rmsnorm=False,
        deepnorm=True,
        apply_residual_connection_post_layernorm=True,
        post_layer_norm=True,
        add_bias_linear=True,
        add_qkv_bias=True,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        rotary_embedding_2d=False,
        rotary_freq_base=10000,
        lora=False,
        mlp_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0,
        use_pytorch_sdpa=True,
        use_flash_attn=False,
        batch_size=1,
        pool_method='attention',
        species_pool_method='mean',
        out_dim=512,
        is_causal=True,
        moe=False,
        num_experts=16, 
        experts_per_token=2,
        **kwargs
    ):

        if not deepnorm and apply_residual_connection_post_layernorm:
            print(f"Warning: deepnorm is False and apply_residual_connection_post_layernorm is True")

        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.torch_dtype = torch_dtype
        self.glu_activation = glu_activation
        self.rmsnorm = rmsnorm
        self.deepnorm = deepnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.rotary_embedding_2d = rotary_embedding_2d
        self.rotary_freq_base = rotary_freq_base
        self.is_causal = is_causal
        self.use_flash_attn = use_flash_attn
        self.batch_size = batch_size
        self.pool_method = pool_method
        self.species_pool_method = species_pool_method
        self.out_dim = out_dim
        self.lora = lora
        self.mlp_lora = mlp_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_pytorch_sdpa = use_pytorch_sdpa
        self.moe = moe
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        super().__init__(**kwargs)