import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional
import json

from services.speech_to_speech.helper import ModelHelper
from config import LLM_SYSTEM_PROMPT


class LLM(ModelHelper):
    def __init__(
        self, 
        model_id: str, 
        torch_dtype: torch.dtype = torch.float32, 
        use_gpu: bool = True,
        quantize: str = 'none',
        use_flash_attn: bool = False
    ) -> None:
        super().__init__(model_id, AutoModelForCausalLM, torch_dtype, use_gpu, use_flash_attn)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.quantize = quantize
        self.system_prompt = LLM_SYSTEM_PROMPT
        
        self.max_messages = 9 
        self.memory: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def load_model(self) -> None:
        if not (self._is_gpu_available and self.use_gpu):
            self.quantize = 'none'
            self.use_flash_attn = False

        quantization_config = None
        if self.quantize == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16, 
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.torch_dtype = torch.float16
            print(f"Applying 4-bit quantization for {self.model_id}")
        elif self.quantize == '8bit':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            print(f"Applying 8-bit quantization for {self.model_id}")
        else:
            print(f"Loading {self.model_id} without quantization.")

        model_kwargs = {
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "device_map": "auto" if self._is_gpu_available and self.use_gpu else "cpu",
        }

        if self._is_gpu_available and self.use_gpu and self.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16
            self.torch_dtype = torch.float16
            print("Enabling Flash Attention 2 (forcing torch.float16 compute dtype)")
        elif self.quantize == 'none':
            model_kwargs["torch_dtype"] = self.torch_dtype
        
        try:
            self.model = self.model_class.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                **model_kwargs
            )
            
            self.model.eval()
            
            if hasattr(self.model, 'device'):
                 self.device = self.model.device
            
            print(f"Model {self.model_id} loaded on {self.device}")
            if self.quantize != 'none':
                print(f"Quantization: {self.quantize}")

            try:
                if torch.__version__ >= "2.0.0" and self.device.type == 'cuda':
                    self.model = torch.compile(self.model)
                    print(f"LLM Model {self.model_id} compiled with torch.compile()")
            except Exception as compile_exc:
                print(f"Warning: torch.compile() failed for LLM: {compile_exc}")

        except Exception as exc:
            print(f"Error loading model {self.model_id}: {exc}")
            if "flash_attention_2" in str(exc):
                print("!!! Flash Attention 2 failed to load. The model may not support it, or your GPU/driver is incompatible. !!!")
            raise

    def run(self, user_input: str, cwd: str, max_new_tokens: int = 256) -> str:
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before run().")

        contextual_user_input = f"CWD: {cwd}\nUser: \"{user_input}\""
        
        self.memory.append({"role": "user", "content": contextual_user_input})

        # Prune older messages, keeping the system prompt and the most recent N turns
        while len(self.memory) > self.max_messages:
            self.memory.pop(1)
            self.memory.pop(1)

        input_ids = self.tokenizer.apply_chat_template(
            self.memory,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = generated_ids[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        response_text = response_text.strip()

        if response_text:
            try:
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1:
                    clean_json_str = response_text[start:end+1]
                    # We still save command responses to memory to maintain context flow
                    # (e.g., "ls -l" is followed by terminal output)
                    self.memory.append({"role": "assistant", "content": clean_json_str})
            except json.JSONDecodeError:
                # If the LLM produces garbage, it's not saved to memory
                print(f"Warning: LLM hallucinated non-JSON, not adding to memory: {response_text}")
                pass 

        return response_text

    def clear_memory(self) -> None:
        print("Clearing conversational memory...")
        self.memory = [{"role": "system", "content": self.system_prompt}]