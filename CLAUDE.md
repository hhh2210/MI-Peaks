# MI-Peaks Project Context

## Project Overview
Research project analyzing mutual information (MI) peaks in Large Reasoning Models (LRMs) during reasoning processes. The project demonstrates that specific "thinking tokens" (like "Hmm", "Wait", "Therefore") correspond to information peaks that are crucial for reasoning performance.

## Key Components

### Core Analysis Scripts
- `src/cal_mi.py` - Calculate mutual information metrics
- `src/CKA.py` - Centered Kernel Alignment analysis
- `src/generate_activation.py` - Generate model activations
- `src/generate_gt_activation.py` - Generate ground truth activations
- `src/mi_estimators.py` - MI estimation implementations

### Applications
- `src/applications/RR_evaluate.py` & `src/applications/RR_model.py` - Representation Recycling method
- `src/applications/TTTS_evaluate.py` - Thinking Token Token Selection evaluation
- `src/applications/evaluate.py` - Main evaluation framework
- `src/applications/grader.py` - Answer grading utilities
- `src/applications/python_executor.py` - Python code execution for math problems

### Data
- `src/applications/data/gsm8k/` - GSM8K dataset (math word problems)
- `src/applications/data/MATH-500/` - MATH benchmark subset
- `src/applications/data/aime24/` - AIME 2024 test problems
- `src/data/math_train_12k.csv` - Training math problems

  generate_activation.py vs generate_gt_activation.py

  | 特征   | generate_activation.py    | generate_gt_activation.py    |
  |------|---------------------------|------------------------------|
  | 输入数据 | math_data['problem'] (问题) | math_data['solution'] (标准答案) |
  | 生成方式 | model.generate() (推理生成)   | model() (直接前向传播)             |
  | 保存路径 | acts/reasoning_evolve/    | acts/gt/                     |
  | 用途   | 捕获模型推理过程的激活               | 获取标准答案的激活作为参考基准              |
## Quick Start Commands
source mi/bin/activate
model_path= /apdcephfs_nj7/share_303407286/models/DeepSeek-R1-Distill-Llama-8B/
### Environment Setup
```bash
# Recommended: Python 3.11.5, PyTorch 2.1.2, Transformers 4.46.1
cd /dockerdata/llm_eval/MI-Peaks/src
```

### Run MI Analysis
```bash
sh scripts/compute_mi_trajectories.sh
```


### Evaluation Methods
```bash
# Representation Recycling
sh scripts/run_RR.sh

# Thinking Token Token Selection  
sh scripts/run_TTTS.sh
```

## TTTS (Thinking Token Token Selection) Code Details

### Overview
TTTS is a test-time scaling method that leverages thinking tokens to improve reasoning performance through a three-phase generation process.

### Core Algorithm (`batch_TTTS_generation` function)

#### Phase 1: Base Generation (lines 141-160)
```python
# Generate initial response with base token budget
sampling_params = SamplingParams(
    max_tokens=base_budget,  # typically 2048-4096 tokens
    temperature=0.0,        # deterministic generation
    stop=stop_words,        # dataset-specific stop words
)
vllm_outputs = llm.generate(prompts, sampling_params)
```

**Key aspects:**
- Uses deterministic generation (temperature=0.0)
- Applies dataset-specific stop words for clean termination
- Base budget configurable via `--max_tokens_per_call` (default: 2048)

#### Phase 2: Thinking Token Continuation (lines 162-187)
```python
# Add thinking token and continue generation
for q, a in zip(prompts, outputs):
    concat_prompt = f'{q} {a} {thinking_token}'  # Simple concatenation
    budget_prompts.append(concat_prompt)

# Generate with additional token budget
sampling_params = SamplingParams(max_tokens=token_budget, ...)
vllm_outputs_budget = llm.generate(budget_prompts, sampling_params)
```

**Key aspects:**
- Thinking token added as plain text (not XML tags)
- Additional token budget via `--token_budget` (default: 2048)
- Common thinking tokens: "Hmm", "Wait", "Therefore", "So", "Let"

#### Phase 3: Final Answer Extraction (lines 189-215)
```python
# Prompt for structured final answer
for q, a in zip(budget_prompts, outputs_budget):
    concat_prompt = q + a + r" Final Answer within \boxed{}:"
    final_prompts.append(concat_prompt)

# Generate final boxed answer (limited tokens)
sampling_params = SamplingParams(max_tokens=16, ...)
final_vllm_outputs = llm.generate(final_prompts, sampling_params)
```

**Key aspects:**
- Forces structured answer format with `\boxed{}` notation
- Very limited token budget (16 tokens) for final answer
- Ensures consistent answer extraction

### Main Execution Flow (`main` function)

#### Data Processing (lines 229-257)
- Loads examples from datasets (GSM8K, MATH, AIME)
- Constructs prompts using `construct_prompt()` 
- Handles few-shot examples and prompt templates
- Supports chat template application via `--apply_chat_template`

#### Multi-Epoch Generation (lines 287-340)
```python
for epoch in range(max_func_call):  # max_func_call = 1 for CoT, 4 for tool-integrated
    outputs = batch_TTTS_generation(...)
    
    # Handle different prompt types
    if args.prompt_type == "pal":
        # Program-Aided Language model
        remain_prompts.append((i, query))
    elif "boxed" not in output and output.endswith("```"):
        # Tool-integrated: execute code blocks
        program = extract_program(query)
        remain_results = executor.batch_apply([program])
    else:
        end_prompts.append((i, query))
```

**Key aspects:**
- Supports iterative generation for tool-integrated prompts
- Handles code execution via `PythonExecutor`
- Different stopping criteria for different prompt types

#### Stop Words Configuration (lines 272-284)
```python
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
if args.prompt_type in ["cot"]:
    stop_words.append("\n\nQuestion:")
if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
    stop_words.extend(["\n\n---", "```output"])
```

### Command Line Arguments

#### TTTS-Specific Parameters
- `--thinking_tokens_file_path`: Path to JSON file containing thinking tokens
- `--thinking_token`: Single thinking token to use (e.g., "Hmm")
- `--max_tokens_per_call`: Base generation budget (default: 2048)
- `--token_budget`: Additional budget for thinking token phase (default: 2048)

#### Model and Generation Parameters
- `--use_vllm`: Use VLLM for parallel generation
- `--temperature`: Generation temperature (default: 0)
- `--n_sampling`: Number of samples per prompt (default: 1)
- `--apply_chat_template`: Apply model's chat template

### Key Implementation Details

#### Thinking Token Processing
- **NOT XML-based**: Thinking tokens are plain text strings, not `<think>` tags
- **Simple concatenation**: Added directly to the end of base generation
- **No special parsing**: Output processed as regular text

#### VLLM Integration
- Parallel batch generation for efficiency
- Custom stop token IDs for specific models (Qwen2: [151645, 151643])
- Request ID sorting to maintain order

#### Answer Extraction and Evaluation
- Multiple choice handling for standardized tests
- Boxed answer extraction for math problems
- Code execution results integration for PAL/tool-integrated prompts

### Evaluation Pipeline
1. Load thinking tokens from JSONL file
2. Filter to alphabetic words only: `re.fullmatch(r'[A-Za-z]+', word)`
3. Iterate through datasets and thinking tokens
4. Run TTTS evaluation for each combination
5. Aggregate results and save metrics

## Development Notes
- Project focuses on information-theoretic analysis of LLM reasoning
- Main contribution: identification of MI peaks corresponding to thinking tokens
- Two proposed methods: Representation Recycling (RR) and Thinking Token Token Selection (TTTS)
- Evaluation on mathematical reasoning benchmarks (GSM8K, MATH, AIME)

---

# RR (Recursive Reasoning) Method Code Analysis

## Overview
RR方法通过在LLM的特定层级对"thinking tokens"进行重复计算来增强推理能力。核心机制是在同一层级提取和注入隐藏状态表征。

## 核心组件

### 1. RecursiveThinkingModel 类 (`RR_model.py`)

#### 初始化参数
```python
class RecursiveThinkingModel(torch.nn.Module):
    def __init__(
        self,
        base_model_name: str = None,
        extract_layer_id: int = None,      # 提取层ID，默认-1（最后一层）
        inject_layer_id: int = None,       # 注入层ID，默认-1（最后一层）
        num_recursive_steps: int = 1,      # 递归步数
        use_recursive_thinking: bool = True,
        output_file: str = None
    )
```

#### 层级识别机制 (`_find_layers`)
```python
def _find_layers(self):
    """支持多种transformer架构"""
    if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
        self.layers = self.base_model.model.layers  # Llama风格
    elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
        self.layers = self.base_model.transformer.h  # GPT风格
    elif hasattr(self.base_model, 'encoder') and hasattr(self.base_model.encoder, 'layer'):
        self.layers = self.base_model.encoder.layer  # BERT风格
```

### 2. 钩子机制 (`_register_hooks`)

#### 提取钩子
```python
def extract_hook(module, inputs, outputs):
    """在指定层提取隐藏状态"""
    if self.enable_extract:
        self.extracted_hidden = outputs[0].clone()  # 克隆输出的隐藏状态
```

#### 注入钩子
```python
def inject_hook(module, inputs):
    """在指定层注入之前提取的隐藏状态"""
    if self.enable_inject and self.extracted_hidden is not None:
        modified_hidden_states = inputs[0].clone()
        # 将最后一个token的隐藏状态替换为提取的状态
        modified_hidden_states[:, -1:, :] = self.extracted_hidden[:, -1:, :]
        return (modified_hidden_states,) + inputs[1:]
    return inputs
```

#### 钩子注册
```python
# 在指定层注册钩子
self.inject_hook = self.layers[self.inject_layer_id].register_forward_pre_hook(inject_hook)
self.extract_hook = self.layers[self.extract_layer_id].register_forward_hook(extract_hook)
```

### 3. 生成流程 (`generate` 方法)

#### 双路径生成机制
1. **常规解码路径**：标准的自回归生成
2. **递归解码路径**：对特定tokens进行重复计算

#### 递归条件判断
```python
# 满足以下条件触发递归解码：
# 1. 启用递归思维
# 2. 候选token在感兴趣的token集合中
# 3. 50%的随机概率
if use_recursive_thinking and candidate_token.item() in interested_tokens and random.random() < 0.5:
    recursive_tokens_count += 1
```

#### 递归前向传播过程
```python
# 1. 启用注入，禁用提取
self.enable_inject = True
self.enable_extract = False

# 2. 执行递归前向传播
recursive_outputs = self.base_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    use_cache=True
)

# 3. 恢复提取，禁用注入
self.enable_extract = True
self.enable_inject = False

# 4. 获取最终token
final_logits = recursive_outputs.logits[:, -1, :]
final_token = final_logits.argmax(dim=-1, keepdim=True)
```

## 关键配置

### 层级配置
- **默认层级**：`-1`（最后一层）
- **实际使用**：Layer 23（基于实验脚本`run_RR.sh`）
- **提取层和注入层**：使用相同的层级ID

### 感兴趣的Tokens
```python
# 从JSONL文件加载英文单词的token IDs
english_word_token_ids = set()
target_word_count = 10  # 选择前10个英文单词

# 读取预处理的token信息
with open(args.interested_tokens_file_path, 'r') as f:
    for line in f:
        record = json.loads(line)
        word = record.get("word", "")
        token_ids = record.get("token_ids", [])
        
        if english_pattern.match(word):  # 匹配英文单词
            for token_id in token_ids:
                english_word_token_ids.add(token_id)
```

## 实验配置 (`run_RR.sh`)

### 模型和数据
```bash
model_path='your_dir/DeepSeek-R1-Distill-Llama-8B'
datasets=("aime24")
ei_layers=(23)  # 使用Layer 23作为提取和注入层
```

### 关键参数
```bash
--inject_layer_id 23
--extract_layer_id 23
--use_recursive_thinking True
--num_recursive_steps 1
--interested_tokens_file_path data/${model_name}.jsonl
```

## 性能统计

### 生成统计
```python
entry = {
    "query": prompt,
    "response": response_str,
    "performance": {
        "recursive_tokens": recursive_tokens_count,    # 递归处理的token数
        "regular_tokens": regular_tokens_count,        # 常规处理的token数  
        "lose_tokens": lose_tokens_count,              # 丢失的token数
    }
}
```

### 内存管理
```python
def _clear_memory(self):
    """清理内存，释放不必要的张量和缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## 技术要点

1. **同层循环**：提取层和注入层使用相同ID，形成表征循环处理
2. **选择性处理**：仅对特定的"感兴趣tokens"进行递归计算
3. **随机触发**：50%概率触发递归，避免所有符合条件的tokens都进入递归
4. **KV缓存更新**：递归后更新键值缓存以保持生成的连续性
5. **内存优化**：及时清理临时变量和GPU缓存

## 文件结构
- `RR_model.py`: 核心递归思维模型实现
- `RR_evaluate.py`: 评估框架和参数处理
- `run_RR.sh`: 实验运行脚本
- `data/${model_name}.jsonl`: 感兴趣tokens的预处理数据

这种方法通过在特定层级重复计算thinking tokens的表征，增强了模型的推理能力，特别是在数学推理任务中表现出色。