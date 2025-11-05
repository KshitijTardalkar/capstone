import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional

from services.speech_to_speech.helper import ModelHelper


class LLM(ModelHelper):
    """
    Large Language Model (LLM) wrapper for conversational AI.
    """

    def __init__(
        self, 
        model_id: str, 
        torch_dtype: torch.dtype = torch.float32, 
        use_gpu: bool = True
    ) -> None:
        """
        Initialize the LLM wrapper.
        """
        super().__init__(model_id, AutoModelForCausalLM, torch_dtype, use_gpu)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.system_prompt = str("you are a helpful assistant. Your output should be given in a single sentence as it will be spoken out by a TTS system.")

        self.memory: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def run(self, user_input: str, max_new_tokens: int = 256) -> str:
        """
        Generate a response to user input using the conversational LLM.
        """
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before run().")

        self.memory.append({"role": "user", "content": user_input})

        # Apply chat template and tokenize
        input_ids = self.tokenizer.apply_chat_template(
            self.memory,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device) # Move *only the input tensor* to the model's device

        # Generate response (model is already on self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = generated_ids[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        response_text = response_text.strip()

        self.memory.append({"role": "assistant", "content": response_text})

        return response_text

    def clear_memory(self) -> None:
        """
        Clear conversation memory and reset to initial system prompt.
        """
        print("Clearing conversational memory...")
        self.memory = [{"role": "system", "content": self.system_prompt}]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        """
        return self.memory[1:]

    def set_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt and reset conversation memory.
        """
        self.system_prompt = new_prompt
        self.memory = [{"role": "system", "content": self.system_prompt}]
        print("System prompt updated and memory cleared.")