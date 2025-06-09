import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from mi_estimators import estimate_mi_hsic


def calculate_mi(acts, gt_acts, layers=[], num_samples=-1, save_dir='results/mi/', args=None):

    num_samples = len(acts) if num_samples < 0 else num_samples
    num_layers = len(acts[0]['reps'])

    mi_list = []

    all_mi_matrices_list = []

    try:
        final_mi_dict = torch.load(f'{save_dir}/{args.dataset}_gtmodel={args.gt_model}_testmodel={args.test_model}.pth')
    except:
        final_mi_dict = {
            id: {
                    'reps': {layer: [] for layer in range(num_layers)},
                    'total_tokens': -1
                }
            for id in range(num_samples)
        }

    for id in tqdm(range(0, num_samples)):  # for each query

        if final_mi_dict[id]['total_tokens'] > 0:
            print(f'id {id} has been computed, skip.')
            continue
        
        final_mi_dict[id]['total_tokens'] = acts[id]['token_ids'].shape[0]


        if len(layers) == 0:
            layers = [layer for layer in acts[id]['reps'].keys()]
        
        layer_mi_matrix = []
        for layer in layers:  
            here_num_tokens = acts[id]['reps'][layer].shape[0]  
            layer_mi_list = torch.zeros(here_num_tokens)

            for i in range(here_num_tokens):
                layer_mi_list[i] =  estimate_mi_hsic(acts[id]['reps'][layer][i], gt_acts[id][layer][0])
                print(f'loop {i} finished!')

            layer_mi_matrix.append(layer_mi_list)
            final_mi_dict[id]['reps'][layer] = layer_mi_list  


            print(f'[id={id}, layer={layer}] layer_mi_list:', layer_mi_list)
            print('total_tokens:', acts[id]['token_ids'].shape[0])
        
        all_mi_matrices_list.append(layer_mi_matrix)

        print(f'id {id} mi_matrix computation finished!')

        # save
        os.makedirs(save_dir, exist_ok=True)
        torch.save(final_mi_dict, f'{save_dir}/{args.dataset}_gtmodel={args.gt_model}_testmodel={args.test_model}.pth')

        
    return final_mi_dict



def load_reps(dataset_name, model_tag, is_gt=False, step_level=False):
    print(f'Loading activations of model [{model_tag}], on dataset [{dataset_name}]...')

    if is_gt:
        return torch.load(f"acts/gt/{dataset_name}_{model_tag}.pth")
    else:
        return torch.load(f"acts/reasoning_evolve/{dataset_name}_{model_tag}.pth")


def main():
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")

    parser.add_argument("--gt_model", default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
    parser.add_argument("--test_model", default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
    parser.add_argument("--dataset", default='math_train_12k')

    parser.add_argument("--sample_num", type=int, default=100)  


    args = parser.parse_args()

    sample_num = args.sample_num


    dataset_name = args.dataset

    gt_model = args.gt_model.split('/')[-1]
    test_model = args.test_model.split('/')[-1]
    layers = args.layers

    acts = load_reps(dataset_name=dataset_name, model_tag=test_model)
    gt_acts = load_reps(dataset_name=dataset_name, model_tag=gt_model, is_gt=True)


    final_mi_dict = calculate_mi(acts=acts, gt_acts=gt_acts, layers=layers, num_samples=args.sample_num, save_dir=save_dir, args=args)
        
    save_dir = f'results/mi'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(final_mi_dict, f'{save_dir}/{dataset_name}_gtmodel={gt_model}_testmodel={test_model}.pth')


if __name__=='__main__':
    main()

