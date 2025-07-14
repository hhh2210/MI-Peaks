import random
import os
import argparse
import time
import re
import json
import torch

from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
from RR_model import RecursiveThinkingModel




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="aime24", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 means use all data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=16000, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template to prompts")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--adapt_few_shot", action="store_true", help="Use few-shot examples for multiple-choice questions, zero-shot for others")
    
    # Add RecursiveThinkingModel specific parameters
    parser.add_argument("--use_recursive_thinking", type=bool, default=True, help="Enable recursive thinking feature")
    parser.add_argument("--extract_layer_id", type=int, default=-1, help="Layer ID for extracting hidden states")
    parser.add_argument("--inject_layer_id", type=int, default=-1, help="Layer ID for injecting hidden states")
    parser.add_argument("--num_recursive_steps", type=int, default=1, help="Number of recursive thinking steps")
    parser.add_argument("--interested_tokens", help="List of token IDs to apply recursive thinking, can be a string or a preprocessed list")
    parser.add_argument("--interested_tokens_file_path", help="interested_tokens_file_path")
    parser.add_argument("--output_file",default="outputs.jsonl", help="Output address for responses")
    args = parser.parse_args()
    args.top_p = (1 if args.temperature == 0 else args.top_p)  # top_p must be 1 when using greedy sampling
    return args



def setup(args):
    """Set up the model and evaluation environment"""
    # Output debug information about interested_tokens
    print(f"Type of interested_tokens: {type(args.interested_tokens)}")
    print(f"Length of interested_tokens: {len(args.interested_tokens) if hasattr(args.interested_tokens, '__len__') else 'N/A'}")
    if args.interested_tokens is not None and hasattr(args.interested_tokens, '__len__') and len(args.interested_tokens) > 0:
        print(f"Elements of interested_tokens: {args.interested_tokens}")
    
    # Load the model
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    
    
    # Load RecursiveThinkingModel
    if args.use_recursive_thinking:
        print("Loading RecursiveThinkingModel...")
        base_model = args.model_name_or_path
        
        # Process the list of interested tokens
        interested_tokens = args.interested_tokens
        print(f"Loaded interested_tokens_ids: Type={type(interested_tokens)}, Length={len(interested_tokens) if interested_tokens else 0}")
        if interested_tokens and len(interested_tokens) > 0:
            print(f"Loaded token IDs: {interested_tokens}")
            
        # Load tokenizer separately for preprocessing
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token    
        # Create RecursiveThinkingModel
        llm = RecursiveThinkingModel(
            base_model_name=base_model,
            extract_layer_id=args.extract_layer_id,
            inject_layer_id=args.inject_layer_id,
            num_recursive_steps=args.num_recursive_steps,
            use_recursive_thinking=args.use_recursive_thinking,
            output_file=args.output_file,
        )
        print(f"RecursiveThinkingModel loaded, extraction layer={args.extract_layer_id}, injection layer={args.inject_layer_id}, recursive steps={args.num_recursive_steps}")
    else:
        # Standard HF model loading
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # Inference and evaluation
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # Add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # Print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def prepare_data(data_name, args):
    """Prepare evaluation data"""
    examples = load_data(data_name, args.split, args.data_dir)

    # Sample num_test_sample samples from the dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # Shuffle data
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # Select start and end indices
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # Get output filename
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    if args.use_recursive_thinking:
        out_file_prefix += f"_recursive{args.num_recursive_steps}_ext{args.extract_layer_id}_inj{args.inject_layer_id}"
        # Add an identifier indicating that interested tokens were used
        if args.interested_tokens is not None:
            out_file_prefix += "_with_interest_tokens"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # Load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # Deduplicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def generate_with_recursive_model(model, prompt, max_tokens, interested_tokens=None, use_recursive_thinking=True):
    """Generate text using RecursiveThinkingModel"""
    return model.generate(
        max_tokens=max_tokens,
        prompt=prompt,
        interested_tokens=interested_tokens,
        use_recursive_thinking=use_recursive_thinking
    )


def is_multi_choice(answer):
    """Check if the answer is in multiple-choice format"""
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    """Main evaluation function"""
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("Data:", data_name, " , Remaining samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # Initialize Python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # Parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # Add remaining fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # Prepare prompts
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # Start inference
    # Measure time usage
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # Get all outputs
        prompts = [item[1] for item in current_prompts]
        
        # Use model to generate outputs
        if hasattr(llm, 'generate') and isinstance(llm, RecursiveThinkingModel):
            # Use RecursiveThinkingModel's generate method
            outputs = []
            # Process the list of interested tokens
            interested_tokens = args.interested_tokens
            
            for prompt in prompts:
                output = generate_with_recursive_model(
                    model=llm, 
                    prompt=prompt, 
                    max_tokens=args.max_tokens_per_call,
                    interested_tokens=interested_tokens,
                    use_recursive_thinking=args.use_recursive_thinking
                )
                # Remove prompt from output, keep only the generated part
                if prompt in output:
                    output = output[len(prompt):]
                outputs.append(output)
                
                # Check for stop words and truncate
                for stop_word in stop_words:
                    if stop_word in output:
                        output = output.split(stop_word)[0]
                        break
        else:
            # Standard HF generation
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # Process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # Execute remaining prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # Not finished
            if epoch == max_func_call - 1:
                query += "\nReached maximum function call limit."
            remain_prompts[k] = (i, query)

    # Unresolved samples
    print("Unresolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # Sort by index
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # Remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # Extract predictions
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # Put results back into samples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # Remove any non-choice characters
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # Add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # Save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minute"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )
    
    # Add recursive thinking parameters to metrics
    if args.use_recursive_thinking:
        result_json["recursive_thinking"] = {
            "enabled": args.use_recursive_thinking,
            "extract_layer_id": args.extract_layer_id,
            "inject_layer_id": args.inject_layer_id,
            "num_recursive_steps": args.num_recursive_steps,
            "interested_tokens_count": len(args.interested_tokens) if args.interested_tokens else 0
        }

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":

    args = parse_args()

    english_word_token_ids = set()
    english_pattern = re.compile(r'^[a-zA-Z]+$')

    english_word_count = 0
    target_word_count = 10

    # Read JSONL file
    with open(args.interested_tokens_file_path, 'r') as f:
        for line in f:
            try:
                if english_word_count >= target_word_count:
                    break

                # Parse JSON
                record = json.loads(line)
                word = record.get("word", "")
                token_ids = record.get("token_ids", [])
                
                if english_pattern.match(word):
                    english_word_count += 1
                    for token_id in token_ids:
                        english_word_token_ids.add(token_id)
                        
            except json.JSONDecodeError:
                print(f"Error: Unable to parse JSON line: {line}")
            except Exception as e:
                print(f"Error during processing: {e}")

    args.interested_tokens = english_word_token_ids

    set_seed(args.seed)
    setup(args)