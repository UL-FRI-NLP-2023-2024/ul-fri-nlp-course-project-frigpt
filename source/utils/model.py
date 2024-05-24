# from google.colab import userdata
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, BitsAndBytesConfig
import torch

HF_Token = "<Input your HF token here>"

def create_text_generator():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # loading in 4 bit
        bnb_4bit_quant_type="nf4", # quantization type
        bnb_4bit_use_double_quant=True, # nested quantization
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HF_Token, torch_dtype=torch.float16, quantization_config=bnb_config)
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HF_Token, torch_dtype=torch.float16)

    text_generator = pipeline('text-generation',
                            model=model,
                            tokenizer=tokenizer,
                            torch_dtype=torch.bfloat16,
                            # device=0,
                            # device_map="auto",
                            do_sample=True,
                            top_k=10,
                            num_return_sequences=1,
                            max_length=5_000,
                            )
    
    return text_generator
