# import torch
# from vllm.model_executor.models.llama import LlamaForCausalLM
# from quip import QUIP
# from qlinear import QuantLinear
# # from codebook import codebook_id
# from codebook.e8p12_rvq4 import E8P12RVQ4B_codebook

# class QuipLlamaForCausalLM(LlamaForCausalLM):
#     def __init__(self, config, **kwargs):
#         super().__init__(config, **kwargs)
#         # self.codebook = codebook_id[config.quantization_config["codebook"]](inference=True)
#         self.codebook = E8P12RVQ4B_codebook(inference=True)
#         self._replace_with_quant_layers()

#     def _replace_with_quant_layers(self):
#         for name, module in self.named_modules():
#             if isinstance(module, torch.nn.Linear):
#                 in_features = module.in_features
#                 out_features = module.out_features
#                 bias = module.bias is not None
#                 new_module = QuantLinear(
#                     in_features,
#                     out_features,
#                     self.codebook,
#                     bias=bias,
#                     use_rand=self.config.quantization_config["use_rand"],
#                     per_channel=self.config.quantization_config["per_channel"]
#                 )
#                 parent_name, child_name = name.rsplit('.', 1)
#                 parent = self.get_submodule(parent_name)
#                 setattr(parent, child_name, new_module)

#     @staticmethod
#     def load_weights(model: "QuipLlamaForCausalLM", checkpoint_path: str):
#         state_dict = torch.load(checkpoint_path, map_location="cpu")
#         for name, param in model.named_parameters():
#             if name in state_dict:
#                 if isinstance(param, QuantLinear):
#                     param.pack(state_dict[name], state_dict[f"{name}_quantizer"])
#                 else:
#                     param.data.copy_(state_dict[name])
#         model.tie_weights()