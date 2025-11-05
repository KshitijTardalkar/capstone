"""
Large Language Model (LLM) service module.

This file defines the `LLM` class, which manages the conversational agent.
It loads a causal LM (like Llama or Gemma) and implements logic for:
- 4-bit or 8-bit quantization (BitsAndBytes).
- `torch.compile()` optimization.
- Maintaining a conversational memory.
- Generating JSON-formatted responses based on a system prompt.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional
import json

from services.speech_to_speech.helper import ModelHelper
from config import LLM_SYSTEM_PROMPT


class LLM(ModelHelper):
    """
    Large Language Model service for conversational AI and tool use.

    Handles loading the LLM (with optional quantization) and generating
    responses. It maintains a chat history and is guided by a system
    prompt to produce structured JSON output.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for the LLM.
        quantize (str): The quantization mode ('none', '4bit', '8bit').
        system_prompt (str): The system prompt defining the agent's behavior.
        memory (List[Dict[str, str]]): The conversational history.
        max_messages (int): The max number of messages (user + assistant)
                            to keep in memory, plus the system prompt.
    """

    def __init__(
        self, 
        model_id: str, 
        torch_dtype: torch.dtype = torch.float32, 
        use_gpu: bool = True,
        quantize: str = 'none',
        use_flash_attn: bool = False
    ) -> None:
        """
        Initializes the LLM service.

        Args:
            model_id (str): The Hugging Face model identifier (e.g., 'meta-llama/Llama-3.2-3B-Instruct').
            torch_dtype (torch.dtype): The desired data type for the model.
            use_gpu (bool): Whether to use the GPU if available.
            quantize (str): The quantization mode ('none', '4bit', '8bit').
            use_flash_attn (bool): Whether to use Flash Attention 2.
        """
        super().__init__(model_id, AutoModelForCausalLM, torch_dtype, use_gpu, use_flash_attn)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.quantize = quantize
        self.system_prompt = LLM_SYSTEM_PROMPT
        
        # Max history: 1 system prompt + 4 turns (4 user + 4 assistant)
        self.max_messages = 9 
        self.memory: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def load_model(self) -> None:
        """
        Loads the LLM from Hugging Face.

        Applies 4-bit or 8-bit quantization if specified and available.
        Applies `torch.compile()` for optimization on GPU.
        """
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
        """
        Generates a model response based on the user input and current directory.

        Implements a sliding window for memory to prevent context overflow.
        Only 'chat' type responses are saved to memory to keep it purely conversational.

        Args:
            user_input (str): The text transcribed from user's speech.
            cwd (str): The current working directory of the terminal.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The raw JSON string output from the LLM.
        
        Raises:
            ValueError: If the model is not loaded before running.
        """
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before run().")

        contextual_user_input = f"CWD: {cwd}\nUser: \"{user_input}\""
        
        self.memory.append({"role": "user", "content": contextual_user_input})

        # --- MEMORY PRUNING LOGIC ---
        # Prune memory if it's too long, *after* adding the new user prompt.
        # This keeps the system prompt (index 0) and the most recent messages.
        while len(self.memory) > self.max_messages:
            # Remove the oldest user/assistant pair (index 1 and 2)
            self.memory.pop(1)
            self.memory.pop(1)
        # --- END MEMORY PRUNING ---

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

        # --- NEW MEMORY LOGIC ---
        # ONLY add the assistant's response to memory if it's a 'chat' response.
        # This prevents command proposals (which are also JSON) or terminal
        # output (which is never seen here anyway) from cluttering the
        # *conversational* history.
        if response_text:
            try:
                # Find the first { and last } to make parsing robust
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1:
                    clean_json_str = response_text[start:end+1]
                    parsed_json = json.loads(clean_json_str)
                    
                    if parsed_json.get("type") == "chat":
                        self.memory.append({"role": "assistant", "content": clean_json_str})
                # If it's a "command" or not valid JSON, we DO NOT add it to memory.
            except json.JSONDecodeError:
                # It was a hallucination (not valid JSON), so don't add it.
                print(f"Warning: LLM hallucinated non-JSON, not adding to memory: {response_text}")
                pass 
        # --- END NEW MEMORY LOGIC ---

        return response_text

    def clear_memory(self) -> None:
        """Resets the conversational memory to just the system prompt."""
        print("Clearing conversational memory...")
        self.memory = [{"role": "system", "content": self.system_prompt}]