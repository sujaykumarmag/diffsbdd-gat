from torch import nn
import torch
import math


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0,
                 reflection_equiv=True):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.reflection_equiv = reflection_equiv
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.cross_product_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer
        ) if not self.reflection_equiv else None
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, coord_cross,
                    edge_attr, edge_mask, update_coords_mask=None):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)

        if not self.reflection_equiv:
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross) * self.coords_range
            trans = trans + coord_cross * phi_cross

        if edge_mask is not None:
            trans = trans * edge_mask

        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        if update_coords_mask is not None:
            agg = update_coords_mask * agg

        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, coord_cross,
                edge_attr=None, node_mask=None, edge_mask=None,
                update_coords_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, coord_cross,
                                 edge_attr, edge_mask,
                                 update_coords_mask=update_coords_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', reflection_equiv=True):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method,
                                                       reflection_equiv=self.reflection_equiv))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None,
                edge_attr=None, update_coords_mask=None, batch_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.reflection_equiv:
            coord_cross = None
        else:
            coord_cross = coord2cross(x, edge_index, batch_mask,
                                      self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr,
                                               node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, coord_cross, edge_attr,
                                       node_mask, edge_mask, update_coords_mask=update_coords_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x

import torch
from torch_scatter import scatter_add
import torch.nn as nn
import torch.nn.functional as F

class GINLayer(nn.Module):
    def __init__(self, in_channels, out_channels, aggregator_type='sum', train_eps=True):
        super(GINLayer, self).__init__()
        self.eps = nn.Parameter(torch.Tensor([0])) if train_eps else None
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.aggregator_type = aggregator_type

    def forward(self, x, edge_index):
        row, col = edge_index
        aggr_out = torch.zeros_like(x)
        aggr_out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        
        if self.eps is not None:
            out = (1 + self.eps) * aggr_out + x
        else:
            out = aggr_out + x
        out = self.mlp(out)
        return out


import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class EGNNHybrid(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum', reflection_equiv=True,
                 use_gin=False, use_gat=False, gat_heads=4, use_mha=False, mha_heads=8):
        super(EGNNHybrid, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv
        self.use_gin = use_gin
        self.use_gat = use_gat
        self.use_mha = use_mha

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        edge_feat_nf = edge_feat_nf + in_edge_nf

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        # Multi-Head Attention layer
        if self.use_mha:
            self.mha = nn.MultiheadAttention(embed_dim=self.hidden_nf, num_heads=mha_heads, batch_first=True)

        # Add GIN layers if specified
        if self.use_gin:
            self.gin_layers = nn.ModuleList([GINLayer(self.hidden_nf, self.hidden_nf) for _ in range(n_layers)])

        # Add GAT layers if specified
        if self.use_gat:
            self.gat_layers = nn.ModuleList([GATConv(self.hidden_nf, self.hidden_nf // gat_heads, heads=gat_heads, concat=True) for _ in range(n_layers)])

        # Add Equivariant Blocks
        for i in range(n_layers):
            self.add_module(f"e_block_{i}", EquivariantBlock(
                hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                act_fn=act_fn, n_layers=inv_sublayers,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range, norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
                reflection_equiv=self.reflection_equiv
            ))

        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, update_coords_mask=None,
                batch_mask=None, edge_attr=None):
        edge_feat, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            edge_feat = self.sin_embedding(edge_feat)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)

        h = self.embedding(h)

        # Apply Multi-Head Attention (MHA) if use_mha is True
        if self.use_mha:
            # Reshape h to (batch_size, num_nodes, hidden_nf)
            h = h.unsqueeze(0) if len(h.shape) == 2 else h
            h, _ = self.mha(h, h, h)
            h = h.squeeze(0)

        if self.use_gin:
            for gin_layer in self.gin_layers:
                h = gin_layer(h, edge_index)

        if self.use_gat:
            for gat_layer in self.gat_layers:
                h = gat_layer(h, edge_index)

        for i in range(self.n_layers):
            h, x = self._modules[f"e_block_{i}"](
                h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask,
                edge_attr=edge_feat, update_coords_mask=update_coords_mask,
                batch_mask=batch_mask
            )

        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum', reflection_equiv=True):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        edge_feat_nf = edge_feat_nf + in_edge_nf

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               reflection_equiv=self.reflection_equiv))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, update_coords_mask=None,
                batch_mask=None, edge_attr=None):
        edge_feat, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            edge_feat = self.sin_embedding(edge_feat)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask,
                edge_attr=edge_feat, update_coords_mask=update_coords_mask,
                batch_mask=batch_mask)

        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv


class GINLayer(nn.Module):
    def __init__(self, in_channels, out_channels, aggr='sum', hidden_channels=None):
        super(GINLayer, self).__init__()
        self.aggr = aggr
        if hidden_channels is None:
            hidden_channels = out_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr=None):
        return GINConv(self.mlp)(x, edge_index)


class GINBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2,
                 normalization_factor=100, aggregation_method='sum'):
        super(GINBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(n_layers):
            self.add_module(f"gin_layer_{i}", GINLayer(in_channels=self.hidden_nf, 
                                                      out_channels=self.hidden_nf,
                                                      aggr=aggregation_method))
        
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        for i in range(self.n_layers):
            h = self._modules[f"gin_layer_{i}"](h, edge_index, edge_attr)
        
        if node_mask is not None:
            h = h * node_mask
        
        return h, x

class GIN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3,
                 out_node_nf=None, normalization_factor=100, aggregation_method='sum'):
        super(GIN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf

        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        edge_feat_nf = in_edge_nf + 2  # Edge feature size adjustment
        
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(n_layers):
            self.add_module(f"gin_block_{i}", GINBlock(hidden_nf, edge_feat_nf=edge_feat_nf, 
                                                       device=device, act_fn=act_fn,
                                                       n_layers=2, normalization_factor=normalization_factor,
                                                       aggregation_method=aggregation_method))
        
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, update_coords_mask=None,
                batch_mask=None, edge_attr=None):
        # Prepare edge features and embeddings
        edge_feat, _ = coord2diff(x, edge_index)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)

        h = self.embedding(h)

        for i in range(self.n_layers):
            h, x = self._modules[f"gin_block_{i}"](h, x, edge_index, node_mask=node_mask,
                                                   edge_mask=edge_mask, edge_attr=edge_feat)
        h = self.embedding_out(h)
        
        if node_mask is not None:
            h = h * node_mask
        if update_coords_mask is not None:
            pass

        return h, x




class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def coord2cross(x, edge_index, batch_mask, norm_constant=1):

    mean = unsorted_segment_sum(x, batch_mask,
                                num_segments=batch_mask.max() + 1,
                                normalization_factor=None,
                                aggregation_method='mean')
    row, col = edge_index
    cross = torch.cross(x[row]-mean[batch_mask[row]],
                        x[col]-mean[batch_mask[col]], dim=1)
    norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    cross = cross / (norm + norm_constant)
    return cross


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
