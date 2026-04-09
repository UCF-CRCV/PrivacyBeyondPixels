import torch.nn as nn


# Self-attention block.
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, input_features):
        output, _ = self.self_attn(input_features, input_features, input_features)
        return output


# Transformer-based anonymizer.
class TransformerAnonymizer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TransformerAnonymizer, self).__init__()
        self.self_attn_blocks = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, input_features):
        output = input_features
        for self_attn_block in self.self_attn_blocks:
            output = self_attn_block(output) + output
        return output
    

# MLP-based anonymizer.
class MLP(nn.Module):
    def __init__(self, initial_embedding_size=768, final_embedding_size=768, use_normalization=False):
        super(MLP, self).__init__()
        self.initial_embedding_size = initial_embedding_size
        self.final_embedding_size = final_embedding_size
        self.fc1 = nn.Linear(self.initial_embedding_size, self.initial_embedding_size, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.initial_embedding_size, self.final_embedding_size, bias=False)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x) # No normalization.
        return x
