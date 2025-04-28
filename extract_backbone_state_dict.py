import torch

if __name__ == "__main__":

    checkpoint_path = './vit_small_checkpoint.pth'
    # load the checkpoint
    checkpoint_state_dict = torch.load(checkpoint_path, map_location='cpu')
    # load the teacher model
    checkpoint_state_dict = checkpoint_state_dict['teacher']
    # remove `module.` prefix
    checkpoint_state_dict = {k.replace("module.", ""): v for k, v in checkpoint_state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    checkpoint_state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint_state_dict.items()}
    # remove `head.` keys
    checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if 'head' not in k}
    # save the new checkpoint
    torch.save(checkpoint_state_dict, checkpoint_path.replace('checkpoint.pth', 'backbone.pth'))
