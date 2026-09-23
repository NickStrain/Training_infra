from torch import nn

class transformer_block(nn.Module):
    def __init__(self,d_model=256, nhead=4,dim_feedforward=512 ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model,dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward,d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        atte_out, _ = self.attn(x,x,x)
        x = self.norm1(atte_out+x)
        x = self.norm2(x+self.ff(x))

        return x 



class Model(nn.Module):
    def __init__(self,vocab_size=1000, d_model=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,d_model)
        self.transformer_layer = nn.ModuleList(
            [transformer_block(d_model) for _ in range(num_layers)]
        )
        self.out = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        x = self.embed(x)
        for layer in self.transformer_layer:
            x = layer(x)

        return self.out(x)

if __name__ == "__main__":
    model = Model()
    num_para = sum(p.numel() for p in model.parameters())
    para_size = sum(p.numel()*p.element_size() for p in model.parameters())/(1024**2)
    print(f'No of parameters {num_para}')
    print(f'Size of the parameters {para_size}')
