import os.path
import torch
from src.chroma.data import Protein
from model.utils import load_VQAE_model, load_VQMaskGPT_model




def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FoldMaskGIT')

    parser.add_argument('--checkpoint', default='/storage/huyuqi/gzy/FoldMaskGPT/model_zoom/params.ckpt', type=str)
    parser.add_argument('--config', default='/storage/huyuqi/gzy/FoldMaskGPT/model_zoom/config.yaml', type=str)
    parser.add_argument('--mask_mode', default='conditional', type=str)
    parser.add_argument('--start_iter', default=0, type=int)
    parser.add_argument('--num_iter', default=20, type=int)
    parser.add_argument('--temperature', default=2.0, type=float)
    parser.add_argument('--length', default=150, type=int)
    parser.add_argument('--nums', default=20, type=int)

    parser.add_argument('--valid_batch_size', default=1, type=int, help='batch size for validation')
    parser.add_argument('--level', default=8, type=int)
    parser.add_argument('--mask_method', default='cosine', type=str)  # 🔍
    parser.add_argument('--save_path', default='/storage/huyuqi/gzy/FoldMaskGPT/results/pred_pdb', type=str)
    parser.add_argument('--template', default='./8vrwB.pdb', type=str)
    parser.add_argument('--mask', default='1-92', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vq_model = load_VQAE_model()
    model, tokenizer = load_VQMaskGPT_model(args.checkpoint, args.config)
    
    
    length = args.length
    nums = args.nums

    with torch.no_grad():
        success_num = 0
        tries = 0

        # # ====================== encoder ========================
        # protein = Protein(file_name, device='cuda')
        # h_V, vq_code, batch_id, chain_encoding = vq_model.encode_protein(protein, args.level)

        # # ====================== MaskGIT ========================
        # vq_ids, label_mask, pos_ids, labels = mask_data(
        #     vq_code,
        #     mode='MaskGIT',
        #     mask_token_id=data_module.testset.tokenizer.vocab['[MLM]'],
        #     mask_method=data_module.hparams["mask_method"],
        # )
        
        if args.mask_mode == 'conditional':
            protein = Protein(args.template, device='cuda') 
            h_V_quat, vq_ids_raw,  batch_id, chain_encoding = vq_model.encode_protein(protein)
            mask_indices = []
            for segment in args.mask.split(','): 
                if '-' in segment:
                    start, end = map(int, segment.split('-'))
                    mask_indices.extend(list(range(start, end)))
                else:
                    mask_indices.append(int(segment))
            vq_ids = vq_ids_raw.clone()
            vq_ids[mask_indices] = tokenizer.vocab['[MLM]']
        else:
            vq_ids = torch.ones(length, dtype=torch.long, device='cuda')
            vq_ids[:] = tokenizer.vocab['[MLM]']

        
        chain_encoding = torch.ones_like(vq_ids)
        pos_ids = torch.arange(vq_ids.shape[0], device='cuda')
        
        
        vq_ids = vq_ids.unsqueeze(0).repeat(nums, 1)
        pos_ids = pos_ids.unsqueeze(0).repeat(nums, 1)

        vq_ids_pred = model.generate(
            vq_ids,
            mask_token_id=tokenizer.vocab['[MLM]'],
            num_iter=args.num_iter,
            start_iter=args.start_iter,
            temperature=args.temperature,
            mask_scheduling_method=args.mask_method,
            position_ids=pos_ids,
            attention_mask=None,
        )
        vq_ids_pred = vq_ids_pred[-1, :, :]
        
        for idx in range(vq_ids_pred.shape[0]):
            # ====================== decoder ========================
            h_V = vq_model.model.vq.embed_id(vq_ids_pred[idx], level=args.level)
            pred_protein = vq_model.model.decoding(h_V, chain_encoding + 1, batch_id=None, returnX=False)
            
            if args.mask_mode == 'unconditional':
                os.makedirs(f'{args.save_path}/temp{args.temperature}/pred_pdb_gpt_{length}/', exist_ok=True)
                pred_protein.to(f'{args.save_path}/temp{args.temperature}/pred_pdb_gpt_{length}/gen_{idx}.pdb')
            else:
                name = args.template.split('.')[-2]
                os.makedirs(f'{args.save_path}/temp{args.temperature}_{args.mask}', exist_ok=True)
                pred_protein.to(f'{args.save_path}/temp{args.temperature}_{args.mask}/{name}_gpt_{idx}.pdb', mask_indices=mask_indices)




