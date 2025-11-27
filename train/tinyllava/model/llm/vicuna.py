# vicuna
from transformers import LlamaForCausalLM, AutoTokenizer

from . import register_llm

@register_llm('vicuna')
def return_vicunaclass():
    def tokenizer_and_post_load(tokenizer):
        # Vicuna 通常基于 Llama，Llama 模型通常没有明确的 pad_token。
        # 在 Fine-tuning 或生成时，经常会设置 eos_token 或 unk_token 为 pad_token。
        # 这里我们假设使用 unk_token 作为 pad_token，如果模型有特定的 pad_token，请调整。
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return (LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load))