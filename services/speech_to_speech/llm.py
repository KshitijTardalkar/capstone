import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional

from services.speech_to_speech.helper import ModelHelper


class LLM(ModelHelper):
    """
    Large Language Model (LLM) wrapper for conversational AI in industrial control rooms.

    Extends `ModelHelper` to provide text generation capabilities using
    pretrained causal language models from Hugging Face. Handles conversation
    management, prompt formatting, and response generation with optional device
    management (CPU/GPU).

    Responsibilities:
    - Initialize and manage conversational LLMs via Hugging Face Transformers.
    - Maintain conversation memory for contextual responses.
    - Format prompts using chat templates for optimal model performance.
    - Generate responses with configurable parameters and safety considerations.
    - Provide GPU-aware inference execution with automatic cleanup.

    Attributes:
        tokenizer (AutoTokenizer): Hugging Face tokenizer for text processing.
        system_prompt (str): System prompt defining the AI assistant's role and behavior.
        memory (List[Dict[str, str]]): Conversation history for maintaining context.
        model (torch.nn.Module, optional): LLM instance loaded via ModelHelper.
    """

    def __init__(self, model_id: str, torch_dtype: torch.dtype = torch.float32) -> None:
        """
        Initialize the LLM wrapper with a given model ID and dtype.

        Args:
            model_id (str): Identifier or local path of the LLM to load.
            torch_dtype (torch.dtype, optional): Desired tensor dtype for model weights
                                                 and inference. Defaults to torch.float32.

        Notes:
            - The model itself is not loaded at construction time. Call `load_model()`
              from the parent `ModelHelper` class before running inference.
            - `tokenizer` is always initialized here for text processing.
            - Conversation memory is initialized with a system prompt.
        """
        super().__init__(model_id, AutoModelForCausalLM, torch_dtype)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Ensure pad token is set for proper tokenization
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.system_prompt: str = (
        #     "You are an expert AI assistant for an industrial control room. "
        #     "Provide clear, concise, and accurate information based on the data provided. "
        #     "Prioritize safety and operational efficiency in all responses. "
        #     "Keep responses brief and actionable for industrial operators."
        # )
        self.system_prompt = str("you are a helpful assistant. Your output should be given in a single sentence as it will be spoken out by a TTS system.")

        self.memory: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def run(self, user_input: str, max_new_tokens: int = 256) -> str:
        """
        Generate a response to user input using the conversational LLM.

        Args:
            user_input (str): User's question or request.
            max_new_tokens (int): Maximum number of tokens to generate in response.
                                 Defaults to 256.

        Returns:
            str: Generated response from the LLM.

        Raises:
            ValueError: If the model has not been loaded (call `load_model()` first).
            Exception: Propagates any errors during tokenization or generation.

        Workflow:
            1. Add user input to conversation memory.
            2. Apply chat template to format the conversation for the model.
            3. Generate response tokens using the model.
            4. Decode the response and add to conversation memory.
            5. Return the generated response text.
        """
        if self.model is None:
            raise ValueError("Model not loaded yet; call load_model() before run().")

        # Add user input to conversation memory
        self.memory.append({"role": "user", "content": user_input})

        # Apply chat template and tokenize
        input_ids = self.tokenizer.apply_chat_template(
            self.memory,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate response
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

        # Extract only the new tokens (response)
        response_ids = generated_ids[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Clean up response text
        response_text = response_text.strip()

        # Add response to conversation memory
        self.memory.append({"role": "assistant", "content": response_text})

        return response_text

    def clear_memory(self) -> None:
        """
        Clear conversation memory and reset to initial system prompt.
        
        Useful for starting fresh conversations or preventing memory buildup
        in long-running sessions.
        """
        print("Clearing conversational memory...")
        self.memory = [{"role": "system", "content": self.system_prompt}]

    def run_and_cleanup(
        self, 
        user_input: str, 
        max_new_tokens: int = 256
    ) -> Optional[str]:
        """
        Generate a response with optional GPU acceleration and safe cleanup.

        Args:
            user_input (str): User's question or request.
            max_new_tokens (int): Maximum number of tokens to generate in response.
                                 Defaults to 256.

        Returns:
            Optional[str]: Generated response if successful, None otherwise.

        Workflow:
            - If a GPU is available, move the model to GPU before inference.
            - Run standard `run()` generation pipeline.
            - On completion (or error), return model to CPU and clear GPU memory.
        
        Notes:
            - Designed for one-off generations where GPU memory should be freed
              immediately after inference.
            - Safe to call repeatedly, as device state is reset at the end.
        """
        if self.has_gpu:
            print("GPU detected. Moving LLM to GPU for inference.")
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

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.

        Returns:
            List[Dict[str, str]]: Current conversation memory excluding system prompt.
        """
        # Return conversation history excluding the system prompt
        return self.memory[1:]  # Skip the first system message

    def set_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt and reset conversation memory.

        Args:
            new_prompt (str): New system prompt to define AI behavior.
        """
        self.system_prompt = new_prompt
        self.memory = [{"role": "system", "content": self.system_prompt}]
        print("System prompt updated and memory cleared.")