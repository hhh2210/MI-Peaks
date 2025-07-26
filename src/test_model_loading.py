import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def test_model_loading():
    """Test model loading with different configurations"""
    model_path = '/apdcephfs_nj7/share_303407286/models/DeepSeek-R1-Distill-Llama-8B'
    
    print(f"Testing model loading from: {model_path}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "No GPU")
    
    # Test 1: Default loading (as in RR_model.py)
    print("\n1. Testing default loading (as in RecursiveThinkingModel):")
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        load_time = time.time() - start_time
        print(f"   Load time: {load_time:.2f} seconds")
        print(f"   Model dtype: {next(model.parameters()).dtype}")
        print(f"   Model device: {next(model.parameters()).device}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Loading with torch_dtype specified
    print("\n2. Testing with torch_dtype=torch.float16:")
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )
        load_time = time.time() - start_time
        print(f"   Load time: {load_time:.2f} seconds")
        print(f"   Model dtype: {next(model.parameters()).dtype}")
        print(f"   Model device: {next(model.parameters()).device}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Loading with device_map="auto"
    print("\n3. Testing with device_map='auto':")
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        load_time = time.time() - start_time
        print(f"   Load time: {load_time:.2f} seconds")
        print(f"   Model dtype: {next(model.parameters()).dtype}")
        print(f"   Model device: {next(model.parameters()).device}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Loading with low_cpu_mem_usage
    print("\n4. Testing with low_cpu_mem_usage=True:")
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        load_time = time.time() - start_time
        print(f"   Load time: {load_time:.2f} seconds")
        print(f"   Model dtype: {next(model.parameters()).dtype}")
        print(f"   Model device: {next(model.parameters()).device}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 5: Loading tokenizer
    print("\n5. Testing tokenizer loading:")
    start_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        load_time = time.time() - start_time
        print(f"   Load time: {load_time:.2f} seconds")
        print(f"   Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_model_loading()