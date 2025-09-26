from helper import ModelHelper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class stt(ModelHelper):
    def __init__(self, model_id, torch_dtype=torch.float32):
        super().__init__(model_id, AutoModelForCausalLM, torch_dtype)
        self.tokenizer= AutoTokenizer.from_pretrained(model_id)

        self.system_prompt: str={
            "You are an expert AI assistant for an industrial control room. "
            "Provide clear, concise, and accurate information. Prioritize safety. "
        }

        self.memory: List[Dict[str, str]]=[
            {
                {"role", "system", "content": self.system_prompt}
            }
        ]
    

    def run(self, user_input:str, mar_new_tokens: int=256)-> str:
        if self.model is None:
            raise ValueError("The model has not been loaded yet. call load_model() before calling.")
        
        self.memory=[{
            "role", "system", "content": user_input
        }]

        inpur_ids=self.tokenizer.apply_chat_template(
            self.memory,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)


        response_ids=generated_ids[0][input_ids.shape[-1]:]
        response_text=self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.memory.append([
            {"role": "assistant", "content": response_text}
        ])

        return response_text
    
    def clear_memory(self)-> None:
        print("Clearning conversational memory")
        self.memory=[{
            "role": "system", "content": self.system_prompt
        }]
    
    def run_and-cleanup(self, user_inputL str, max_new_tokens: int=256) -> Optional[str]:
        if self.has_gpu:
            self.to_gpu(dtype=self.torch_dtype)

        response = None
        try:
            response = self.run(user_input, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"An error occurred during LLM inference: {e}")
        finally:
            if self.device.type == "cuda":
                self.to_cpu(dtype=self.torch_dtype)
                print("LLM moved back to CPU and GPU memory cleared.")
        
        return response



        