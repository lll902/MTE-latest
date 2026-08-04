"""
__coding__: utf-8
__Author__: Liu Zhihan
__Time__: 2024/8/21 14:30
__File__: MTEGDRP.py
__remark__:
__Software__: PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import ReLU
import torch.nn.functional as F
from torch_geometric.nn import GINConv as GIN_layer
from torch_geometric.nn import GCNConv as GCN_layer
from torch_geometric.nn import GATConv as GAT_layer, BatchNorm, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch import nn, einsum
from einops import rearrange
from models.egnn_pytorch import EGNN

DIST_KERNELS = {
    'exp': {
        'fn': lambda t: torch.exp(-t),
        'mask_value_fn': lambda t: torch.finfo(t.dtype).max
    },
    'softmax': {
        'fn': lambda t: torch.softmax(t, dim=-1),
        'mask_value_fn': lambda t: -torch.finfo(t.dtype).max
    }
}


def exists(val):
    return val is not None


def default(val, d):
    return d if not exists(val) else val


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim=78, heads=6, dim_head=13, Lg=0.5, Ld=0.5, La=1, dist_kernel_fn='exp'):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.La = La
        self.Ld = Ld
        self.Lg = Lg

        self.dist_kernel_fn = dist_kernel_fn

    def forward(self, x, mask=None, adjacency_mat=None, distance_mat=None):
        h, La, Ld, Lg, dist_kernel_fn = self.heads, self.La, self.Ld, self.Lg, self.dist_kernel_fn

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (h qkv d) -> b h n qkv d', h=h, qkv=3).unbind(dim=-2)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(distance_mat):
            distance_mat = rearrange(distance_mat, 'b i j -> b () i j')

        if exists(adjacency_mat):
            adjacency_mat = rearrange(adjacency_mat, 'b i j -> b () i j')

        if exists(mask):
            mask_value = torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]

            # mask attention
            dots.masked_fill_(~mask, -mask_value)

            if exists(adjacency_mat):
                adjacency_mat.masked_fill_(~mask, 0.)

        attn = dots.softmax(dim=-1)

        # sum contributions from adjacency and distance tensors
        attn = attn * La

        if exists(adjacency_mat):
            attn = attn + Lg * adjacency_mat

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MAT(nn.Module):
    def __init__(
            self,
            *,
            dim_in=78,
            model_dim=78,
            dim_out=78,
            depth=1,
            heads=6,
            Lg=0.5,
            Ld=0.5,
            La=1,
            dist_kernel_fn='exp'
    ):
        super().__init__()

        self.embed_to_model = nn.Linear(dim_in, model_dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            layer = nn.ModuleList([
                Residual(PreNorm(model_dim, Attention(model_dim, heads=heads, Lg=Lg, Ld=Ld, La=La,
                                                      dist_kernel_fn=dist_kernel_fn))),
                Residual(PreNorm(model_dim, FeedForward(model_dim)))
            ])
            self.layers.append(layer)

        self.norm_out = nn.LayerNorm(model_dim)
        self.ff_out = FeedForward(model_dim, dim_out)

    def forward(
            self,
            x,
            mask=None,
            adjacency_mat=None,
            distance_mat=None
    ):
        x = self.embed_to_model(x)

        for (attn, ff) in self.layers:
            x = attn(
                x,
                mask=mask,
                adjacency_mat=adjacency_mat,
                distance_mat=distance_mat
            )
            x = ff(x)

        x = self.norm_out(x)
        x = x.mean(dim=-2)
        x = self.ff_out(x)
        return x



# Define the Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, 2048)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(2048, d_model)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


# Define the Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return self.norm(tgt)


class OmicsTokenEmbedding(nn.Module):
    """Convert each scalar latent omics feature into a learnable token.

    Input shape:  [batch, num_features]
    Output shape: [batch, num_features, model_dim]

    The scalar projection is shared across feature positions, while the
    component embedding preserves the identity/order of each PCA/KPCA feature.
    """

    def __init__(self, num_features, model_dim, dropout=0.2):
        super().__init__()
        self.num_features = num_features
        # nn.Linear(in_features, out_features)的in_features对应输入向量的最后一个特征维度
        self.value_projection = nn.Linear(1, model_dim)
        self.component_embedding = nn.Parameter(
            torch.zeros(1, num_features, model_dim)
        )
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.component_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(
                f'Omics input must have shape [batch, features], got {tuple(x.shape)}'
            )
        if x.size(1) != self.num_features:
            raise ValueError(
                f'Expected {self.num_features} omics features, got {x.size(1)}'
            )

        x = self.value_projection(x.unsqueeze(-1))
        # x.unsqueeze(-1) shape [batch, num_features, 1]
        # x shape: [batch, num_features, model_dim]
        x = x + self.component_embedding
        x = self.norm(x)
        return self.dropout(x)


class OmicsSelfAttentionBlock(nn.Module):
    """A lightweight pre-norm multi-head self-attention block."""

    def __init__(self, model_dim, nhead, ff_dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        #norm1和norm2能生成两套不同归一化缩放参数和偏移参数，所以不能复用

        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, attn_out_weights = self.self_attn(
            norm_x, norm_x, norm_x, need_weights=False
        )
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class OmicsCrossAttention(nn.Module):
    """Retrieve source-omics context for the mRNA query representation."""

    def __init__(self, model_dim, nhead, dropout=0.2):
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.source_norm = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, source):
        query = self.query_norm(query)
        source = self.source_norm(source)
        context, context_weights = self.cross_attn(
            query=query,
            key=source,
            value=source,
            need_weights=False
        )
        return self.dropout(context)

class OmicsAttentionPooling(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.Tanh(),
            nn.Linear(model_dim // 2, 1)
        )

    def forward(self, x):
        # x: [batch, num_tokens, model_dim]
        weights = torch.softmax(self.score(x), dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled

class GatedOmicsResidualFusion(nn.Module):
    """Inject methylation and mutation contexts into the mRNA main branch."""
    def __init__(self, model_dim, dropout=0.2):
        super().__init__()
        self.meth_gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid()
        )
        self.mut_gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, mrna, meth_context, mut_context):
        meth_gate = self.meth_gate(torch.cat([mrna, meth_context], dim=-1))
        mut_gate = self.mut_gate(torch.cat([mrna, mut_context], dim=-1))

        fused = (
            mrna
            + meth_gate * self.dropout(meth_context)
            + mut_gate * self.dropout(mut_context)
        )
        fused = self.norm(fused)

        gate_stats = {
            "meth_mean": meth_gate.mean().detach(),
            "meth_std": meth_gate.std().detach(),
            "mut_mean": mut_gate.mean().detach(),
            "mut_std": mut_gate.std().detach()
        }

        return fused, gate_stats



class MTEGDRP(torch.nn.Module):
    def __init__(self, output_dim=1, num_features_xd=78,
                 ge_features_dim=128, num_features_xt=25, embed_dim=128,
                 mut_feature_dim=128, meth_feature_dim=128, connect_dim=128, dropout=0.2,
                 omics_model_dim=64, omics_nhead=4, omics_ff_dim=128):
        super(MTEGDRP, self).__init__()

        self.mat = MAT(
            dim_in=78,
            model_dim=78,
            dim_out=78 * 2,
            depth=4,
            heads=6,
            Lg=0.5,
            Ld=0.5,
            La=1,
            dist_kernel_fn='exp')
        self.conv_gcn = GCN_layer(num_features_xd * 2, num_features_xd * 2)

        net1 = Sequential(Linear(num_features_xd * 2, num_features_xd * 2), ReLU(),
                          Linear(num_features_xd * 2, num_features_xd * 2))
        self.conv_gin1 = GIN_layer(net1)
        self.bn1 = torch.nn.BatchNorm1d(num_features_xd * 2)

        self.fc_mat_egnn = Linear(num_features_xd * 2, num_features_xd)
        self.drug_layer1 = EGNN(dim=78)
        self.drug_layer2 = EGNN(dim=78)
        self.drug_layer3 = EGNN(dim=156)
        self.fc_drug_jihe1 = Linear(78,390)
        self.fc_drug_jihe2 = Linear(390,156)

        self.fc3_drug = Linear(312,156)

        self.fc1_drug = Linear(num_features_xd * 6, num_features_xd * 12)
        self.fc2_drug = Linear(num_features_xd * 12,num_features_xd * 6)


        # 多组学潜在特征编码与渐进式融合
        # 输入仍沿用现有数据处理结果：[batch, 128]（默认）。
        # 每个潜在特征标量被映射为一个token，使多头注意力沿潜在特征维工作。
        if omics_model_dim % omics_nhead != 0:
            raise ValueError(
                'omics_model_dim must be divisible by omics_nhead, '
                f'got {omics_model_dim} and {omics_nhead}'
            )

        self.ge_token_embedding = OmicsTokenEmbedding(
            ge_features_dim, omics_model_dim, dropout
        )
        self.meth_token_embedding = OmicsTokenEmbedding(
            meth_feature_dim, omics_model_dim, dropout
        )
        self.mut_token_embedding = OmicsTokenEmbedding(
            mut_feature_dim, omics_model_dim, dropout
        )

        # 三种组学分别进行一次多头自注意力
        self.ge_self_attention_1 = OmicsSelfAttentionBlock(
            omics_model_dim, omics_nhead, omics_ff_dim, dropout
        )
        self.meth_self_attention_1 = OmicsSelfAttentionBlock(
            omics_model_dim, omics_nhead, omics_ff_dim, dropout
        )
        self.mut_self_attention_1 = OmicsSelfAttentionBlock(
            omics_model_dim, omics_nhead, omics_ff_dim, dropout
        )

        # 甲基化和突变分别通过交叉注意力向mRNA主支路注入信息
        self.meth_to_ge_cross_attention = OmicsCrossAttention(
            omics_model_dim, omics_nhead, dropout
        )
        self.mut_to_ge_cross_attention = OmicsCrossAttention(
            omics_model_dim, omics_nhead, dropout
        )

        # 门控残差融合
        self.omics_gated_fusion = GatedOmicsResidualFusion(
            omics_model_dim, dropout
        )

        # 融合后的mRNA再进行一次多头自注意力
        self.ge_self_attention_2 = OmicsSelfAttentionBlock(
            omics_model_dim, omics_nhead, omics_ff_dim, dropout
        )

        # 池化后仍输出connect_dim维，保持后续药物-多组学拼接与解码器不变
        self.ge_attention_pool = OmicsAttentionPooling(
            omics_model_dim
        )
        self.ge_output_projection = nn.Sequential(
            nn.LayerNorm(omics_model_dim),
            nn.Linear(omics_model_dim, connect_dim)
        )

        # drug_data维度 = num_features_xd * 6，ge_data维度 = connect_dim。
        fusion_input_dim = num_features_xd * 6 + connect_dim

        # Define the Transformer Decoder
        self.decoder = TransformerDecoder(
            num_layers=4,
            d_model=fusion_input_dim,  # Concatenated feature dimensions
            nhead=4,  # Number of attention heads
            dropout=0.1
        )

        self.fc1_all = Linear(fusion_input_dim, 1024)
        self.fc2_all = Linear(1024, 512)
        self.fc3_all = Linear(512, 256)
        self.fc4_all = Linear(256, 128)
        self.out = Linear(connect_dim, output_dim)

        # 激活函数和正则化
        self.relu = ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        drug_poi_data, drug_edg_index, batch = data.x, data.edge_index, data.batch
        ge_data, meth_data, mut_data = data.target_ge, data.target_meth, data.target_mut
        coors = data.coordinates

        drug_data = torch.unsqueeze(drug_poi_data, 1)
        mat_drug_data = self.mat(drug_data)
        drug_data = self.conv_gcn(mat_drug_data, drug_edg_index)
        drug_data = self.relu(drug_data)
        drug_data = F.relu(self.conv_gin1(drug_data, drug_edg_index))
        drug_data = self.bn1(drug_data)
        drug_data = self.conv_gcn(drug_data, drug_edg_index)
        drug_data = self.relu(drug_data)

        egnn_list =[]
        index = 0
        for data_len in data.c_size:
            data_len = int(data_len.item())
            temp_data = mat_drug_data[index:index+data_len]
            temp_data = self.fc_mat_egnn(temp_data).unsqueeze(0)
            temp_coors = coors[index:index+data_len].unsqueeze(0)
            temp_drug_data_jihe, temp_ehnncoors = self.drug_layer1(temp_data, temp_coors)
            temp_drug_data_jihe, temp_ehnncoors = self.drug_layer2(temp_drug_data_jihe,temp_ehnncoors)
            temp_drug_data_jihe = temp_drug_data_jihe.squeeze(0)
            egnn_list.append(temp_drug_data_jihe)
            index+=data_len

        egnn_features = torch.cat(egnn_list,dim=0)

        drug_data_final = torch.cat((drug_data,egnn_features),dim=1)
        drug_data_final = torch.cat([gmp(drug_data_final, batch), gap(drug_data_final, batch)], dim=1)


        drug_data = self.relu(self.fc1_drug(drug_data_final))
        drug_data = self.dropout(drug_data)
        drug_data = self.fc2_drug(drug_data)

        # -------------------------------------------------------------
        # 多组学潜在特征渐进式融合
        # 1) 三种组学分别进行一次多头自注意力
        ge_tokens = self.ge_token_embedding(ge_data)
        meth_tokens = self.meth_token_embedding(meth_data)
        mut_tokens = self.mut_token_embedding(mut_data)

        ge_tokens = self.ge_self_attention_1(ge_tokens)
        meth_tokens = self.meth_self_attention_1(meth_tokens)
        mut_tokens = self.mut_self_attention_1(mut_tokens)

        # 2) 甲基化和突变分别交叉注意力注入mRNA
        meth_context = self.meth_to_ge_cross_attention(
            query=ge_tokens,
            source=meth_tokens
        )
        mut_context = self.mut_to_ge_cross_attention(
            query=ge_tokens,
            source=mut_tokens
        )

        # 3) 门控残差融合
        fused_ge_tokens, gate_stats = self.omics_gated_fusion(
            mrna=ge_tokens,
            meth_context=meth_context,
            mut_context=mut_context
        )

        # 4) 融合后的mRNA再做一次多头自注意力
        fused_ge_tokens = self.ge_self_attention_2(
            fused_ge_tokens
        )

        # 5) 对融合后的mRNA潜在token进行注意力池化
        ge_summary = self.ge_attention_pool(
            fused_ge_tokens
        )

        # 得到最终细胞系表示
        ge_data = self.ge_output_projection(
            ge_summary
        )

        # -------------------------------------------------------------
        concat_data = torch.cat((drug_data, ge_data), 1)
        # Pass the concatenated features through the Transformer Decoder
        concat_data = concat_data.unsqueeze(0)  # Add batch dimension for Transformer input
        concat_data = self.decoder(concat_data, concat_data)
        concat_data = concat_data.squeeze(0)  # Remove batch dimension after processing



        # 隐藏层
        concat_data = self.fc1_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        concat_data = self.fc2_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        concat_data = self.fc3_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        concat_data = self.fc4_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        out = self.out(concat_data)
        #out = self.sigmoid(out)
        #out = nn.Sigmoid()(out)
        return out, drug_data, ge_data
