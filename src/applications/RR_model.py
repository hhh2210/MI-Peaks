import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List
import gc
import json
import time
import random


class RecursiveThinkingModel(torch.nn.Module):
    def __init__(
            self,
            base_model_name: str = None,
            extract_layer_id: int = None,
            inject_layer_id: int = None,
            num_recursive_steps: int = 1,  # Number of recursive optimization steps
            use_recursive_thinking: bool = True,
            output_file: str = None
    ):
        super().__init__()


        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.extract_layer_id = extract_layer_id or -1
        self.inject_layer_id = inject_layer_id or -1
        self.extracted_hidden = None

        self.num_recursive_steps = num_recursive_steps
        self.use_recursive_thinking = use_recursive_thinking


        self._find_layers()


        self.extract_hook = None
        self.inject_hook = None
        self.enable_inject = False
        self.enable_extract = True


        self.output_file = output_file
        if output_file is None:
            raise ValueError("output_file must be provided as a valid path string")
            

    def _find_layers(self):
        """Identify the layer structure of the model"""
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            self.layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
            self.layers = self.base_model.transformer.h
        elif hasattr(self.base_model, 'encoder') and hasattr(self.base_model.encoder, 'layer'):
            self.layers = self.base_model.encoder.layer
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.base_model)}")

    def _register_hooks(self):
        """Register feature passing hooks"""

        # Remove existing hooks (if any)
        self._remove_hooks()

        # Feature extraction hook (captures representation at the specified layer)
        def extract_hook(module, inputs, outputs):
            if self.enable_extract:
                self.extracted_hidden = outputs[0].clone()

        # Feature injection hook (injects captured representation into the target layer)
        def inject_hook(module, inputs):
            if self.enable_inject and self.extracted_hidden is not None:
                modified_hidden_states = inputs[0].clone()
                modified_hidden_states[:, -1:, :] = self.extracted_hidden[:, -1:, :]

                return (modified_hidden_states,) + inputs[1:]

            return inputs

        
        # Register hooks
        self.inject_hook = self.layers[self.inject_layer_id].register_forward_pre_hook(inject_hook)
        self.extract_hook = self.layers[self.extract_layer_id].register_forward_hook(extract_hook)

    def _remove_hooks(self):
        """Remove hooks to prevent memory leaks"""
        if hasattr(self, 'extract_hook') and self.extract_hook is not None:
            self.extract_hook.remove()
            self.extract_hook = None

        if hasattr(self, 'inject_hook') and self.inject_hook is not None:
            self.inject_hook.remove()
            self.inject_hook = None

    def _clear_memory(self):
        """Clean up memory, release unnecessary tensors and cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @torch.inference_mode()
    def generate(
            self,
            max_tokens: int,
            prompt: str,
            interested_tokens: set = None,
            use_recursive_thinking: bool = True,
            **kwargs
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device).eval()

        # Register hooks only when needed
        if use_recursive_thinking:
            self._register_hooks()

        try:
            # Initialize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            past_key_values = None
            generated = input_ids.clone()
            full_input_ids = input_ids.clone()  # Save the complete input sequence
            print(f"Prompt: {prompt}")
            
            recursive_tokens_count = 0
            regular_tokens_count = 0
            lose_tokens_count = 0
            
            for _ in tqdm(range(max_tokens), desc="Generating"):
                # Regular generation to get candidate token

                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False
                )
                next_token_logits = outputs.logits[:, -1, :]
                # 只取概率最高的token
                candidate_token = next_token_logits.argmax(dim=-1, keepdim=True)
                del next_token_logits

                if candidate_token.item() == self.tokenizer.eos_token_id:
                    break

                # recursive decoding 
                if use_recursive_thinking and candidate_token.item() in interested_tokens and random.random() < 0.5:
                    recursive_tokens_count += 1
                    print(f"Recursive token")

                    final_token = None

                    # ------------------------------ Forward propagation after replacing representation ------------------------------
                    self.enable_inject = True
                    self.enable_extract = False
                    self.base_model.config.use_cache=True

                    recursive_outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    self.enable_extract = True
                    # Disable hooks, reset hidden state
                    self.enable_inject = False
                    final_logits = recursive_outputs.logits[:, -1, :]
                    final_token = final_logits.argmax(dim=-1, keepdim=True)


                    # ------------------------------ Update KV cache ------------------------------
                    past_key_values = recursive_outputs.past_key_values
                    # Update sequence
                    generated = torch.cat([generated,final_token], dim=-1)
                    full_input_ids = torch.cat([full_input_ids,final_token], dim=-1)
                    input_ids = final_token

                    # Update attention_mask
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), device=device)
                    ], dim=1)
                # regular decoding
                else:
                    regular_tokens_count += 1
                    past_key_values = outputs.past_key_values
                    # Update sequence
                    generated = torch.cat([generated, candidate_token], dim=-1)
                    full_input_ids = torch.cat([full_input_ids, candidate_token], dim=-1)
                    input_ids = candidate_token

                    # Update attention_mask
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), device=device)
                    ], dim=1)


                outputs = None

                if _ % 5000 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


            # Save performance statistics
            response_str = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            entry = {
                "query": prompt,
                "response": response_str,
                "performance": {
                    "recursive_tokens": recursive_tokens_count,
                    "regular_tokens": regular_tokens_count,
                    "lose_tokens": lose_tokens_count,

                }
            }
            with open(self.output_file, "a") as f:
                f.write(json.dumps(entry, indent=2) + "\n")

            print(f"\nGenerated {recursive_tokens_count} recursive tokens and {regular_tokens_count} regular tokens")
            print('-'*50)

            generated = None
            full_input_ids = None
            past_key_values = None
            input_ids = None
            attention_mask = None


            return response_str

        finally:
            self._remove_hooks()
            self._clear_memory()