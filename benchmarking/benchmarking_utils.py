import os

def set_benchmarking_path(args):
    if args.hook_module == 'unet':
        return f'results/results_seed_{args.seed}' + '/' + args.res_path.split('/')[2]
    elif args.hook_module == 'text':
        return f'results_CLIP/results_seed_{args.seed}' + '/' + args.res_path.split('/')[2]
    elif args.hook_module == 'unet-ffn-1':
        return f'results_FFN-1/results_seed_{args.seed}' + '/' + args.res_path.split('/')[2]
    elif args.hook_module == 'attn_key':
        return f'results_attn_key/results_seed_{args.seed}' + '/' + args.res_path.split('/')[2]
    elif args.hook_module == 'attn_val':
        return f'results_attn_val/results_seed_{args.seed}' + '/' + args.res_path.split('/')[2]