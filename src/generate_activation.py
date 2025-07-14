import torch
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class Hook:
    def __init__(self, token_position=-1):
        self.token_position = token_position
        self.tokens_embeddings = []

    def __call__(self, module, module_inputs, module_outputs):
        # output: [batch, seq_len, hidden_size]
        hidden_states = module_outputs[0] if isinstance(module_outputs, tuple) else module_outputs
        emb = hidden_states[0, self.token_position].detach().cpu()

        self.tokens_embeddings.append(emb)


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto") 
    
    return tokenizer, model


def get_acts(query_list, tokenizer, model, layers, token_pos=-1):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.

    token_pos: default to fetch the last token's activations
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook(token_position=token_pos)
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    
    # get activations
    acts = {
        id: {
                'reps': {layer: [] for layer in layers},
                'token_ids': []
            }
        for id in range(len(query_list))
    } 


    for id, query in tqdm(enumerate(query_list), total=len(query_list), desc="Processing Queries"):

        input_ids = tokenizer.encode(query, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,  # greedy
                return_dict_in_generate=True,
                output_hidden_states=True
            )

        response = tokenizer.batch_decode(outputs[0][:, input_ids.shape[1]:-1])[0].strip()

        for layer, hook in zip(layers, hooks):
            acts[id]['reps'][layer] = hook.tokens_embeddings
        
        acts[id]['token_ids'] = outputs[0][:, input_ids.shape[1]:-1].squeeze().cpu()  
        
        for hook in hooks:
            hook.tokens_embeddings = []

    for id, layer_acts in acts.items():
        for layer, emb in layer_acts['reps'].items():
            layer_acts['reps'][layer] = torch.stack(emb).float()


    # remove hooks
    for handle in handles:
        handle.remove()
    
    return acts


def main():
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model_path", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--layers", nargs='+', help="Layers to save embeddings from")
    parser.add_argument("--dataset", default='math_train_12k')
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--output_dir", default="acts", help="Directory to save activations to")
    args = parser.parse_args()

    dataset = args.dataset

    math_data = pd.read_csv(f'data/{dataset}.csv')

    query_list = math_data['problem'].tolist()[:args.sample_num]

    tokenizer, model = load_model(args.model_path)

    layers = [int(layer) for layer in args.layers]
    if layers == [-1]:
        layers = list(range(len(model.model.layers)))

    acts = get_acts(query_list, tokenizer, model, layers, token_pos=-1)

    # save representations
    os.makedirs(f'{args.output_dir}/reasoning_evolve/', exist_ok=True)
    torch.save(acts, f"{args.output_dir}/reasoning_evolve/{dataset}_{args.model_path.split('/')[-1]}.pth")


if __name__=='__main__':
    main()

