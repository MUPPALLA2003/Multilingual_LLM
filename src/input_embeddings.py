import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self,vocab_size:int,embed_dim:int) -> None:

        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size,embed_dim)

    def forward(self,ids:torch.Tensor) -> torch.Tensor:

        x = self.tok_emb(ids)

        return x

            

