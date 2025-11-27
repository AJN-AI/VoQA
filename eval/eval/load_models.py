# Load Model
# Ensure that the model path can be imported correctly
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_model(args):
    model_components = {}
    # You can call your loading model functions here. There are some examples as follows:
    # 1. TinyLLaVA
    if args.model_name in ['TinyLLaVA-Phi-2-SigLIP-3.1B', 'TinyLLaVA-Qwen2-0.5B-SigLIP', 'TinyLLaVA-Qwen2.5-3B-SigLIP']:
        from models.TinyLLaVA_Factory.tinyllava_model import load_tinyllava_model
        model, tokenizer, text_processor, image_processor, context_len = load_tinyllava_model(args)
        model_components.update({
            "model": model,
            "tokenizer": tokenizer,
            "text_processor": text_processor,
            "image_processor": image_processor,
            "context_len": context_len,
        })

    # 2. TinyLLaVA for SFT
    elif args.model_name in ['TinyLLaVA-Qwen2-0.5B-SigLIP-Baseline', 'TinyLLaVA-Qwen2-0.5B-SigLIP-QA',  \
                             'TinyLLaVA-Qwen2-0.5B-SigLIP-QRA', 'TinyLLaVA-Qwen2.5-3B-SigLIP-QRA', \
                             'TinyLLaVA-Qwen2-0.5B-SigLIP-QRA-HELPER', 'TinyLLaVA-Qwen2-0.5B-SigLIP-QRA-CAT',
                             'TinyLLaVA-Qwen2-0.5B-SigLIP-R-QRA', 'TinyLLaVA-Qwen2-0.5B-SigLIP-QA-only']:
        from models.TinyLLaVA_Factory_SFT.tinyllava_sft_model import load_tinyllava_sft_model
        model, image_processor, text_processor, tokenizer, context_len = load_tinyllava_uni_model(args)
        model_components.update({
            "model": model,
            "tokenizer": tokenizer,
            "text_processor": text_processor,
            "image_processor": image_processor,
            "context_len": context_len,
        })

    # 3.InternVL2.5   
    elif args.model_name in ['InternVL2_5-1B']:
        from models.InternVL.internvl_model import load_internvl_model
        model, tokenizer, generation_config = load_internvl_model(args)
        model_components.update({
            "model": model,
            "tokenizer": tokenizer,
            "generation_config": generation_config,
        })

    # 4. Internvl3 / Internvl3-SFT
    elif args.model_name in ['InternVL3-1B', 'InternVL3-1B-QA', 'InternVL3-1B-QRA', 'InternVL3-1B-R-QRA', 'InternVL3-1B-QA-only']:
        from models.InternVL.internvl3_model import load_internvl3_model
        model, tokenizer = load_internvl3_model(args)
        model_components.update({
            "model": model,
            "tokenizer": tokenizer
        })

    # 5. LLaVA
    elif args.model_name in ['llava-v1.5-7b']:
        from models.LLaVA.llava_model import load_llava_model
        tokenizer, model, image_processor, context_len, args = load_llava_model(args)
        model_components.update({
            "model": model,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "context_len": context_len,
        })
    
    # 6. LLaVA for SFT/ llava-hf
    elif args.model_name in ['llava-1.5-7b-hf', 'llava-v1.5-7b-Baseline', 'llava-v1.5-7b-QA', 'llava-v1.5-7b-QRA']:
        from models.LLaVA.llava_hf_model import load_llava_hf_model
        model, processor = load_llava_hf_model(args)
        model_components.update({
            "model": model,
            "processor": processor,
        })

    # 7. Qwen2.5-VL
    elif args.model_name in ['Qwen2.5-VL-3B-Instruct']:
        from models.Qwen.qwen_model import load_qwen_model
        model, processor = load_qwen_model(args)
        model_components.update({
            "model": model,
            "processor": processor,
        })

    # 8. Qwen2-VL / Qwen2-VL for SFT
    elif args.model_name in ['Qwen2-VL-2B', 'Qwen2-VL-2B-Baseline', 'Qwen2-VL-2B-QA', 'Qwen2-VL-2B-QRA', 'Qwen2-VL-2B-R-QRA', 'Qwen2-VL-2B-QA-only']:
        from models.Qwen.qwen2_model import load_qwen2_model
        model, processor = load_qwen2_model(args)
        model_components.update({
            "model": model,
            "processor": processor,
        })

    # 9. deepseek
    elif args.model_name in ['deepseek-vl2-tiny']:
        from models.DeepSeek_VL2.deepseek_model import load_deepseek_model
        vl_chat_processor, tokenizer, model = load_deepseek_model(args)
        model_components.update({
            "model": model,
            "tokenizer": tokenizer,
            "vl_chat_processor": vl_chat_processor,
        })

    # 10. BLIP-3
    elif args.model_name in ['xgen-mm-phi3-mini-instruct-interleave-r-v1.5']:
        from models.LAVIS_xgen_mm.blip3_model import load_blip3_model
        model, tokenizer, image_processor = load_blip3_model(args)
        model_components.update({
            "model": model,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
        })
        
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    return model_components   
