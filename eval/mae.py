import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from vit_pytorch.vit import Transformer, ViT
import pytorch_lightning as pl
class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img, masked_patch_idx=None):
        device = img.device
        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens
        tokens = self.patch_to_emb(patches)

        #add positions
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        
        #insert manually selected patches to mask
        if masked_patch_idx is not None:
            assert len(masked_patch_idx) == num_masked, f'wrong number of masked patches chosen, expected {num_masked} {masked_patch_idx}'
            unmasked_patch_idx = torch.tensor([idx for idx in range(num_patches) if idx not in masked_patch_idx], requires_grad=False, device=device)
            masked_indices = repeat(masked_patch_idx, 'd -> b d', b=batch)
            unmasked_indices = repeat(unmasked_patch_idx, 'd -> b d', b=batch)
            
        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        #add cls tokens
        cls_tokens = repeat(self.encoder.cls_token, '1 1 d -> b 1 d', b = batch)
        cls_tokens = cls_tokens + self.encoder.pos_embedding[:, 0] #add position embedding to cls tokens
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        #temporarily remove cls tokens
        cls_tokens = decoder_tokens[:,0]
        decoder_tokens = decoder_tokens[:,1:]

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices+1) #plus 1 to account for cls token being index 0
        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices+1)

        #add position embedding to cls tokens
        cls_indices = torch.zeros((batch,1), dtype=torch.int64, device=device)
        cls_tokens = cls_tokens.unsqueeze(axis=1) + self.decoder_pos_emb(cls_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.zeros(batch, num_patches+1, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices+1] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices+1] = mask_tokens
        decoder_tokens[batch_range, cls_indices] = cls_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices+1]
        pred_pixel_values = self.to_pixels(mask_tokens)

        return masked_patches, pred_pixel_values
    
    
class IF_MAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.mae =  MAE(encoder = ViT(image_size=160,
                                      patch_size=32,
                                      num_classes=1000,
                                      dim=1024,
                                      depth=6,
                                      heads=8,
                                      channels=1,
                                      mlp_dim=2048),
                        masking_ratio = 0.75,    # the paper recommended 75% masked patches
                        decoder_dim = 512,       # paper showed good results with just 512
                        decoder_depth = 12)       # anywhere from 1 to 8
        
    def forward(self, x, masked_patch_idx):
        masked_patches, pred_pixel_values = self.mae(x, masked_patch_idx=masked_patch_idx)
        return masked_patches, pred_pixel_values


if __name__ == '__main__':
    from vit_pytorch import ViT
    x = torch.rand(2,1,160,160,)
    mae = MAE(encoder = ViT(image_size=160,
                              patch_size=32,
                              num_classes=1000,
                              dim=1024,
                              depth=6,
                              heads=8,
                              channels=1,
                              mlp_dim=2048),
                masking_ratio = 0.75,    # the paper recommended 75% masked patches
                decoder_dim = 512,       # paper showed good results with just 512
                decoder_depth = 6)   
    mae(x)
