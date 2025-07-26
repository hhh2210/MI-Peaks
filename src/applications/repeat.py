import random
import os
import argparse
import time
import json
import re
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import LogitsProcessor, AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--save_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1: all data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--apply_chat_template", action="store_true",)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--adapt_few_shot", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)

    # Repeat generation related
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--token_budget", default=2048, type=int)
    parser.add_argument("--repeat_prompt", default="question", type=str)
    parser.add_argument("--continuation_prompt", default="look back the question again, now i need to", type=str)
    args = parser.parse_args()

    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def load_model_and_tokenizer(args):
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )
    return llm, tokenizer


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    examples = examples[args.start: (len(examples) if args.end == -1 else args.end)]

    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    output_dir = os.path.join("outputs", output_dir)
    os.makedirs(os.path.join(output_dir, data_name), exist_ok=True)
    out_file = os.path.join(output_dir, data_name, f"{out_file_prefix}_s{args.start}_e{args.end}.jsonl")

    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f for f in os.listdir(os.path.join(output_dir, data_name))
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(os.path.join(output_dir, data_name, f))))

    processed_samples_dict = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = set(processed_samples_dict.keys())
    examples = [example for example in examples if example["idx"] not in processed_idxs]

    return examples, list(processed_samples_dict.values()), out_file


def setup(args):
    llm, tokenizer = load_model_and_tokenizer(args)
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    avg_acc = sum(result["acc"] for result in results) / len(results)
    data_list.append("avg")
    results.append({"acc": avg_acc})

    pad = max(len(name) for name in data_list)
    print("\t".join(name.ljust(pad, " ") for name in data_list))
    print("\t".join(f"{result['acc']:.1f}".ljust(pad, " ") for result in results))


def is_multi_choice(answer):
    return all(c in ["A", "B", "C", "D", "E"] for c in answer)


# ------------------------------ Repeat Generation with Question Reminder ----------------------------------
def batch_repeat_generation(prompts, llm, tokenizer, args, stop_words):
    base_budget = args.max_tokens_per_call
    token_budget = args.token_budget

    ### 1. Base generation
    sampling_params = SamplingParams(
        max_tokens=base_budget,
        top_p=args.top_p,
        stop=stop_words,
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in args.model_name_or_path.lower()
            else None
        ),
        skip_special_tokens=False,
        temperature=0.0,
    )
    vllm_outputs = llm.generate(prompts, sampling_params)

    outputs = []
    for output in vllm_outputs:
        generated_text = output.outputs[0].text.strip()
        outputs.append(generated_text)

    
    ### 2. Generation with repeat prompt
    repeat_prompts = []

    for q, a in zip(prompts, outputs):
        # Construct repeat prompt with question reminder
        concat_prompt = f'{q} {a} {args.repeat_prompt} {q} {args.continuation_prompt}'
        repeat_prompts.append(concat_prompt)

    sampling_params = SamplingParams(
        max_tokens=token_budget,
        top_p=args.top_p,
        stop=stop_words,
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in args.model_name_or_path.lower()
            else None
        ),
        skip_special_tokens=False,
        temperature=0.0,
    )
    vllm_outputs_repeat = llm.generate(repeat_prompts, sampling_params)
    
    outputs_repeat = []
    for output in vllm_outputs_repeat:
        generated_text = output.outputs[0].text.strip()
        outputs_repeat.append(generated_text)


    ### 3. Final output
    final_prompts = []
    for q, a in zip(repeat_prompts, outputs_repeat):
        concat_prompt = q + a +  r" Final Answer within \boxed{}:"
        final_prompts.append(concat_prompt)


    sampling_params = SamplingParams(
        max_tokens=16,
        top_p=args.top_p,
        stop=stop_words,
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in args.model_name_or_path.lower()
            else None
        ),
        skip_special_tokens=False,
        temperature=0.0,
    )
    final_vllm_outputs = llm.generate(final_prompts, sampling_params)

    final_outputs = []
    for output in final_vllm_outputs:
        generated_text = output.outputs[0].text.strip()
        final_outputs.append(generated_text)

    return final_vllm_outputs


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("Dataset:", data_name, "Total samples:", len(examples))
    if examples:
        print(examples[0])

    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]
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
        for key in [
            "level", "type", "unit", "solution_type", "choices", "solution",
            "ques_type", "ans_type", "answer_type", "dataset", "subfield", "filed",
            "theorem", "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    input_prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]
    if args.apply_chat_template and tokenizer is not None:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = list(enumerate(input_prompts))
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

    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if not current_prompts:
            break

        prompts = [item[1] for item in current_prompts]

        outputs = batch_repeat_generation(
            prompts=prompts,
            llm=llm,
            tokenizer=tokenizer,
            args=args,
            stop_words=stop_words,
        )
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        outputs = [output.outputs[0].text for output in outputs]

        assert len(outputs) == len(current_prompts)

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

        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    end_prompts.extend(remain_prompts)
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    results = [run_execute(executor, code, args.prompt_type, data_name) for code in codes]
    time_use = time.time() - start_time

    all_samples = []
    for i, sample in enumerate(samples):
        code_list = codes[i * args.n_sampling: (i + 1) * args.n_sampling]
        result_list = results[i * args.n_sampling: (i + 1) * args.n_sampling]
        preds = [item[0] for item in result_list]
        reports = [item[1] for item in result_list]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in ["A", "B", "C", "D", "E"]:
                preds[j] = choice_answer_clean(code_list[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                preds[j] = "".join([c for c in preds[j] if c in ["A", "B", "C", "D", "E"]])
        sample.pop("prompt", None)
        sample.update({"code": code_list, "pred": preds, "report": reports})
        all_samples.append(sample)

    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )
    out_file_metrics = out_file.replace(".jsonl", f"_{args.prompt_type}_repeat_metrics.json")
    with open(out_file_metrics, "w") as f:
        json.dump(result_json, f, indent=4)
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    result_json["repeat_prompt"] = args.repeat_prompt
    result_json["continuation_prompt"] = args.continuation_prompt
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    try:
        setup(args)
    except Exception as e:
        print(f"Error during setup: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure proper cleanup for distributed processes
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        
        # Give time for all processes to finish
        import time
        time.sleep(2)