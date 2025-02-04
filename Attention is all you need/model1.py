import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x).float() * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, :x.shape[1], :]
        
        # Set requires_grad to False (as it's a fixed tensor)
        pe.requires_grad = False

        # Add the positional encoding to the input
        x = x + pe
        return self.dropout(x)



class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean( dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        # The dimension of each head should be d_model/h
        self.d_k = d_model // h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # Calculate attention scores
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
            
        
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        # query = self.w_q(q)  # (batch_size, seq_len, d_model)
        # # key = self.w_k(k)    # (batch_size, seq_len, d_model)
        # value = self.w_v(v)  # (batch_size, seq_len, d_model)
        # key = self.w_k(k.float())
        query = self.w_q(q.float())
        key = self.w_k(k.float())
        value = self.w_v(v.float())
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        # query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # if mask is not None:
        #     # Same mask for all heads
        #     mask = mask.unsqueeze(1)
        
        # Apply attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine heads
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # Final linear projection
        return self.w_o(x)
        
 

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attn_block = self_attention_block
        self.feed_forward = feed_forward
        self.residual_conn = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Modified to pass the entire attention computation as a lambda
        x = self.residual_conn[0](x, lambda normalized: self.self_attn_block(normalized, normalized, normalized, src_mask))
        x = self.residual_conn[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList,):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention = self_attention_block
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, enc_mask, dec_mask):
        x = self.residual_connection[0](x, lambda normalized: self.self_attention(normalized, normalized, normalized, dec_mask))
        x = self.residual_connection[1](x, lambda normalized: self.cross_attention(normalized, encoder_output, encoder_output, enc_mask))
        x = self.residual_connection[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, enc_mask, dec_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, enc_mask, dec_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        print(f"Projection layer input shape: {x.shape}")
        return torch.log_softmax(self.proj(x), dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, enc_emb: InputEmbedding, dec_emb: InputEmbedding, enc_pos: PositionalEncoding, dec_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_emb = enc_emb
        self.dec_emb = dec_emb
        self.enc_pos = enc_pos
        self.dec_pos = dec_pos
        self.projection_layer = projection_layer

    def encode(self, enc_input, enc_mask):
        enc = self.enc_emb(enc_input)
        enc = self.enc_pos(enc)
        return self.encoder(enc, enc_mask)

    def decode(self, dec_input, dec_mask, encoder_output, enc_mask):
        dec = self.dec_emb(dec_input)
        dec = self.dec_pos(dec)
        return self.decoder(dec, encoder_output, enc_mask, dec_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(enc_vocab_size: int, dec_vocab_size: int, enc_seq_len: int, dec_seq_len: int, d_model: int = 512, layers: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # Encoder
    enc_embed = InputEmbedding(d_model, enc_vocab_size)
    # Decoder
    dec_embed = InputEmbedding(d_model, dec_vocab_size)

    # Positional Encoding
    enc_pos = PositionalEncoding(d_model,enc_seq_len,  dropout)
    dec_pos = PositionalEncoding(d_model,dec_seq_len,  dropout)

    # Encoder Blocks
    encoder_blocks = [
        EncoderBlock(
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout
        ) for _ in range(layers)
    ]
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # Decoder Blocks
    decoder_blocks = [
        DecoderBlock(
            MultiHeadAttentionBlock(d_model, h, dropout),
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout
        ) for _ in range(layers)
    ]
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Projection Layer
    projection_layer = ProjectionLayer(d_model, dec_vocab_size)

    # Transformer
    transformer = TransformerBlock(encoder, decoder, enc_embed, dec_embed, enc_pos, dec_pos, projection_layer)

    # Initialize weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer