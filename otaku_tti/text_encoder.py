from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, logging as hf_logging
hf_logging.set_verbosity_error()

class CLIPTextEncoder(nn.Module):

    def __init__(self, model_name: str='openai/clip-vit-base-patch32', max_length: int=77, device: Union[str, torch.device]='cpu'):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self._device = torch.device(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.embed_dim = self.text_model.config.hidden_size
        self.to(self._device)
        if self._device.type == 'cuda':
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.text_model.to(dtype=dtype)

    def train(self, mode: bool=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, prompts: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]
        device = next(self.text_model.parameters()).device
        tokens = self.tokenizer(prompts, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sentence_embed = F.normalize(outputs.pooler_output.float(), dim=-1)
        token_embeds = F.normalize(outputs.last_hidden_state.float(), dim=-1)
        return (sentence_embed, token_embeds, attention_mask)

    def forward(self, prompts: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode(prompts)