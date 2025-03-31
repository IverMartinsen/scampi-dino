import torch
import vit_mae as vits_mae

def load_vit_mae_model():

    def interpolate_pos_embed(model, checkpoint_model):
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
    
    model = vits_mae.__dict__['vit_base_patch16'](num_classes=0, drop_path_rate=0.1, global_pool=True)
    
    url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
    checkpoint1 = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    checkpoint1.keys()
    checkpoint = torch.load('/Users/ima029/Downloads/mae_pretrain_vit_base.pth', map_location="cpu")
    checkpoint.keys()
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    
    
    
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    interpolate_pos_embed(model, checkpoint_model)
    
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Missing keys when loading pretrained weights: {msg.missing_keys}")
    
    #if global_pool:
    #    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    #else:
    #    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    
    #trunc_normal_(model.head.weight, std=2e-5)
