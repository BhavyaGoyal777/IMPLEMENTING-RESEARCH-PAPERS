import torch
import torch.nn as nn 
from torchvision import datasets,transforms
from torch.utils.data import DataLoader , random_split



##here the images are given in the shape B,channels,height,width
class PatchEmbeddings(nn.Module):
    def __init__(self,num_channels = 3,
                 patch_size = 16,
                 embedding_dim = 768):
        super().__init__()

        self.patch_size = patch_size
        self.patched_embeddings = nn.Conv2d(in_channels= num_channels,out_channels=embedding_dim,kernel_size=patch_size,stride=patch_size,padding=0)
        self.flatten_embeddings= nn.Flatten(2,3)

                                            
    def forward(self,x):
        image_resolution = x.shape[-1] #used to check the comaptability with the patch size 
        assert image_resolution % self.patch_size == 0

        x_patched = self.patched_embeddings(x)
        x_flatten = self.flatten_embeddings(x_patched) #shape->(batch,768,4)

        return x_flatten.permute(0,2,1)





class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads = 12 ,
                 embedding_dim = 768,
                 attention_dropout = 0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mha = nn.MultiheadAttention(embedding_dim,num_heads,attention_dropout,batch_first = True)
    def forward(self,x):
        x = self.layer_norm(x)
        context_vec,_= self.mha(key = x, value = x , query = x,need_weights = False) # the need wieghts is the attention score which we dont want

        return context_vec
    



class FFN(nn.Module):
    def __init__(self,in_embedding_dim = 768,
                 hidden_emb_size = 3072,
                 dropout = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(in_embedding_dim,hidden_emb_size,bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_emb_size,out_features=in_embedding_dim,bias=False),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        x = self.layer_norm(x)
        x = self.ffn(x)

        return x     






class encoder_block(nn.Module):
    def __init__(self, num_heads =12 , 
                 embedding_dim = 768,
                 drop_out = 0.1,
                 hidden_emb = 3072,
                 attn_drop = 0
                 ):
        super().__init__()

        self.mha_layer = MultiHeadAttention(embedding_dim=embedding_dim,num_heads=num_heads,attention_dropout=drop_out)
        self.ffn = FFN(in_embedding_dim=embedding_dim,hidden_emb_size=hidden_emb,dropout=drop_out)

    def forward(self,x):
        x = self.mha_layer(x) + x #residual connections
        x = self.ffn(x) + x ##residual connections

        return x   




class ViT(nn.Module):
    def __init__(self,
                 embedding_size = 768,
                 patch_size = 16,
                 num_heads = 12,
                 hidden_emb = 3072,
                 dropout = 0.1,
                 attn_dropout = 0,
                 image_height = 224,
                 image_Width = 224,
                 num_channels = 3,
                 classes = 1000,
                 pos_drop = 0.1,
                 num_block = 12

                 ):
        super().__init__()
        self.num_patches = image_height * image_Width // (patch_size * patch_size)
        self.patch_embedding =  PatchEmbeddings(patch_size=patch_size,embedding_dim=embedding_size,num_channels=3)
        self.positional_embedding = nn.Parameter(torch.rand(1,self.num_patches+1,embedding_size),requires_grad = True)
        self.cls_token = nn.Parameter(torch.rand(1,1,embedding_size),requires_grad = True)
        self.positional_drop_out = nn.Dropout(p = pos_drop)


        self.encoder_block = nn.Sequential(
            *[encoder_block(
                num_heads=num_heads,
                embedding_dim=embedding_size,
                hidden_emb=hidden_emb,
                drop_out=attn_dropout) for _ in range(num_block)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_size,
                      out_features=classes)
        )

    def forward(self,x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(batch_size,-1,-1)#okay so here we first reate the cls token embedding to match our batch size so that it can be appended to evry data in the batch
        x = torch.cat((x,cls_token),dim=1)
        x = x + self.positional_embedding

        x = self.positional_drop_out(x)
        x =self.encoder_block(x)
        x = self.classifier(x[:,-1])

        return x 
    

    




 




