import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import RelGraphConv
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.util import similarity, KgeLoss, rat
from dgl.nn.pytorch import GATConv
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from pytorch_pretrained_bert.modeling import BertEncoder, BertConfig, BertLayerNorm, BertPreTrainedModel
from functools import partial
from kge.util import sc
from kge.util import mixer2
import math

class SubgraphGNN(nn.Module):
    def __init__(self, dim, num_relations, num_layers=3, num_heads=4):
        super().__init__()
        # RelGraphConv branch
        self.rel_layers = nn.ModuleList([
            RelGraphConv(dim, dim, num_rels=num_relations, regularizer=None, self_loop=True)
            for _ in range(num_layers)
        ])
        # Graph Attention branch (relation-agnostic)
        self.gat_layers = nn.ModuleList([
            GATConv(in_feats=dim, out_feats=dim // num_heads, num_heads=num_heads, allow_zero_in_degree=True)
            for _ in range(num_layers)
        ])
        self.relu = nn.ReLU()
        # Fuse two branches
        self.fuse_linear = nn.Linear(dim * 2, dim)
        
        # Temporal weight integration
        self.temporal_gate = nn.Sequential(
            nn.Linear(dim + 1, dim),  # +1 for temporal weight
            nn.Sigmoid()
        )

    def forward(self, g):
        h = g.ndata['feat']
        
        # Get temporal weights if available
        temporal_weights = g.edata.get('temporal_weight', None)
        
        for layer_idx, (rel_layer, gat_layer) in enumerate(zip(self.rel_layers, self.gat_layers)):
            h_rel = rel_layer(g, h, g.edata['etype'])             # [N, dim]
            h_gat = gat_layer(g, h).flatten(1)                    # [N, dim]
            h_combined = self.fuse_linear(torch.cat([h_rel, h_gat], dim=1))
            
            # Apply temporal weighting if available
            if temporal_weights is not None and layer_idx == 0:  # Apply only in first layer
                # Aggregate temporal weights to nodes (simple mean for now)
                # This is a simplified approach - in practice, you'd want more sophisticated aggregation
                try:
                    # Create a simple temporal gating mechanism
                    avg_temporal_weight = temporal_weights.mean().unsqueeze(0).expand(h_combined.size(0), 1)
                    temporal_input = torch.cat([h_combined, avg_temporal_weight], dim=1)
                    temporal_gate = self.temporal_gate(temporal_input)
                    h_combined = h_combined * temporal_gate
                except:
                    # Fallback if temporal weighting fails
                    pass
            
            h = self.relu(h_combined)
        return h

class SURGEScorer(RelationalScorer):
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.dim = self.get_option("entity_embedder.dim")
        self.max_context_size = self.get_option("max_context_size")
        self.initializer_range = self.get_option("initializer_range")
        # Use the device specified in config instead of hardcoded device
        self.device = torch.device(self.config.get("job.device"))

        # Dynamic temporal subgraph parameters
        self.base_time_window = self.get_option("base_time_window")  # Base time window size
        self.max_time_window = self.get_option("max_time_window")   # Maximum adaptive window
        self.temporal_decay = self.get_option("temporal_decay")    # Temporal decay factor
        self.max_hops = self.get_option("max_temporal_hops")         # Maximum temporal hops
        self.adaptive_window = self.get_option("adaptive_window") # Enable adaptive windowing
        
        # Performance optimization caches
        self._neighbor_cache = {}  # Cache for temporal neighbors
        self._window_cache = {}    # Cache for adaptive windows
        self._subgraph_cache = {}  # Cache for complete subgraphs
        self.cache_size_limit = 10000  # Limit cache size to prevent memory issues
        
        # Pre-load and cache neighbor data for faster access
        self._preload_neighbor_data()

        self.cls = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.cls, std=self.initializer_range)
        self.global_cls = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.global_cls, std=self.initializer_range)
        self.local_mask = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.local_mask, std=self.initializer_range)
        self.type_embeds = nn.Embedding(100, self.dim)
        torch.nn.init.normal_(self.type_embeds.weight, std=self.initializer_range)
        self.atomic_type_embeds = nn.Embedding(4, self.dim)
        torch.nn.init.normal_(self.atomic_type_embeds.weight, std=self.initializer_range)

        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.dim, 
            num_heads=self.get_option("nhead"),
            dropout=self.get_option("attn_dropout"),
            batch_first=True
        )
        
        # Time window predictor for adaptive windowing
        self.window_predictor = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),  # entity + time embeddings
            nn.ReLU(),
            nn.Linear(self.dim, 1),
            nn.Sigmoid()
        )

        self.similarity = getattr(similarity, self.get_option("similarity"))(self.dim)
        self.layer_norm = BertLayerNorm(self.dim, eps=1e-12)
        self.atomic_layer_norm = BertLayerNorm(self.dim, eps=1e-12)

        self.mixer_encoder = mixer2.MLPMixer(
            num_ctx=self.max_context_size + 1,
            dim=self.dim,
            depth=self.get_option("nlayer"),
            ctx_dim=self.get_option("ff_dim"),
            dropout=self.get_option("attn_dropout")
        )

        self.glb_avg_pooling = nn.AdaptiveAvgPool1d(1)

        config = BertConfig(0, hidden_size=self.dim,
                            num_hidden_layers=self.get_option("nlayer") // 2,
                            num_attention_heads=self.get_option("nhead"),
                            intermediate_size=self.get_option("ff_dim"),
                            hidden_act=self.get_option("activation"),
                            hidden_dropout_prob=self.get_option("hidden_dropout"),
                            attention_probs_dropout_prob=self.get_option("attn_dropout"),
                            max_position_embeddings=0,
                            type_vocab_size=0,
                            initializer_range=self.initializer_range)
        self.atom_encoder = BertEncoder(config)
        self.atom_encoder.config = config
        self.atom_encoder.apply(partial(BertPreTrainedModel.init_bert_weights, self.atom_encoder))

        self.diffusion_steps = 1000
        self.beta = torch.linspace(1e-4, 0.02, self.diffusion_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # sinusoidal time embedding → FiLM
        self.time_embed_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )

        # denoiser now takes [x_t , p_emb, t_emb , time_hidden]  = 4*dim
        self.denoiser = nn.Sequential(
            nn.Linear(4 * self.dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.dim)  # only predict epsilon for e_emb
        )

        # Initialize GNN for dynamic temporal subgraphs
        self.gnn = SubgraphGNN(self.dim, self.dataset.num_relations() * 2).to(self.device)

    def _preload_neighbor_data(self):
        """Pre-load neighbor data to device for faster access"""
        try:
            ctx_list, ctx_size = self.dataset.index('neighbor')
            # Ensure data is on the correct device
            self.ctx_list_cached = ctx_list.to(self.device)
            self.ctx_size_cached = ctx_size.to(self.device)
            self.neighbor_data_available = True
            print(f"Neighbor data preloaded to device: {self.device}")
        except Exception as e:
            self.neighbor_data_available = False
            print(f"Warning: Could not preload neighbor data: {e}. Dynamic subgraphs may be slower.")

    def _get_cache_key(self, entity_id, query_time, time_window):
        """Generate cache key for temporal neighbors"""
        return (entity_id.item(), query_time.item(), time_window.item())

    def _clear_cache_if_needed(self):
        """Clear cache if it gets too large"""
        if len(self._neighbor_cache) > self.cache_size_limit:
            # Clear oldest half of cache (simple LRU approximation)
            keys_to_remove = list(self._neighbor_cache.keys())[:len(self._neighbor_cache)//2]
            for key in keys_to_remove:
                del self._neighbor_cache[key]

    def _ensure_device_consistency(self, *tensors):
        """Ensure all tensors are on the model's device"""
        return tuple(tensor.to(self.device) if tensor is not None else None for tensor in tensors)

    def _add_noise(self, x, t):
        alpha_bar = self.alpha_bar.to(x.device)
        alpha_bar_t = alpha_bar[t].view(-1, 1)
        epsilon = torch.randn_like(x)
        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * epsilon, epsilon

    def _get_timestep_embedding(self, t):
        """Sinusoidal embedding similar to Transformer positional enc."""
        half_dim = self.dim // 2
        device = t.device
        emb = torch.exp(torch.arange(half_dim, device=device) * (-math.log(10000.0) / (half_dim - 1)))
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def _denoise(self, x_t, t, cond):
        # cond = concat[p_emb, t_emb]
        t_embed = self._get_timestep_embedding(t)
        t_hidden = self.time_embed_mlp(t_embed)
        inp = torch.cat([x_t, cond, t_hidden], dim=1)
        return self.denoiser(inp)

    def _compute_adaptive_time_window(self, entity_emb, time_emb, query_time):
        """Compute adaptive time window based on entity and temporal context"""
        if not self.adaptive_window:
            return torch.full((entity_emb.size(0),), self.base_time_window, 
                            device=entity_emb.device, dtype=torch.long)
        
        # Batch processing for efficiency
        batch_size = entity_emb.size(0)
        
        # Combine entity and time embeddings
        combined_emb = torch.cat([entity_emb, time_emb], dim=1)
        
        # Predict window size multiplier (0-1) - batch processing
        with torch.no_grad():  # No gradients needed for window prediction during inference
            window_multiplier = self.window_predictor(combined_emb).squeeze(-1)
        
        # Scale to actual window size
        window_sizes = self.base_time_window + (self.max_time_window - self.base_time_window) * window_multiplier
        return window_sizes.long()

    def _extract_temporal_neighbors(self, entity_id, query_time, time_window):
        """Extract neighbors within temporal window with multi-hop expansion"""
        device = self.device  # Use the model's device consistently
        
        # Ensure all inputs are on the correct device
        entity_id, query_time, time_window = self._ensure_device_consistency(entity_id, query_time, time_window)
        
        # Check cache first
        cache_key = self._get_cache_key(entity_id, query_time, time_window)
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]
        
        # Use pre-loaded data if available
        if self.neighbor_data_available:
            ctx_list = self.ctx_list_cached
            ctx_size = self.ctx_size_cached
        else:
            # Fallback to loading data
            try:
                ctx_list, ctx_size = self.dataset.index('neighbor')
                ctx_list = ctx_list.to(device)
                ctx_size = ctx_size.to(device)
            except Exception:
                return torch.empty((0, 3), device=device, dtype=torch.long), torch.empty((0,), device=device)
        
        # Get neighbors for this entity
        num_neighbors = ctx_size[entity_id]
        if num_neighbors == 0:
            result = (torch.empty((0, 3), device=device, dtype=torch.long), torch.empty((0,), device=device))
            self._neighbor_cache[cache_key] = result
            return result
        
        # Efficient slicing - avoid creating intermediate tensors
        neighbor_data = ctx_list[entity_id, :num_neighbors, :]  # [num_neighbors, 3]
        neighbor_times = neighbor_data[:, 2].long()
        

        
        # Vectorized temporal filtering
        time_diff = torch.abs(neighbor_times - query_time)
        temporal_mask = time_diff <= time_window
        
        # Early termination if no neighbors in window
        if temporal_mask.sum() == 0:
            # Take closest temporal neighbors (limit to 3 for efficiency)
            k = min(3, len(time_diff))
            _, closest_indices = torch.topk(-time_diff, k)
            temporal_mask = torch.zeros_like(temporal_mask, dtype=torch.bool)
            temporal_mask[closest_indices] = True
        
        # Filter neighbors efficiently
        filtered_neighbors = neighbor_data[temporal_mask]
        
        # Compute temporal weights efficiently
        filtered_times = neighbor_times[temporal_mask]
        time_distances = torch.abs(filtered_times - query_time).float()
        temporal_weights = torch.exp(-self.temporal_decay * time_distances)
        
        # Cache result
        result = (filtered_neighbors, temporal_weights)
        self._clear_cache_if_needed()
        self._neighbor_cache[cache_key] = result
        
        return result

    def _expand_temporal_subgraph(self, seed_entities, query_times, time_windows):
        """Multi-hop temporal subgraph expansion - optimized version"""
        device = self.device  # Use the model's device consistently
        
        # Ensure all inputs are on the correct device
        seed_entities = seed_entities.to(device)
        query_times = query_times.to(device)
        time_windows = time_windows.to(device)
        batch_size = seed_entities.size(0)
        
        all_subgraph_data = []
        all_temporal_weights = []
        
        # Process in smaller batches to balance memory and speed
        for i in range(batch_size):
            entity_id = seed_entities[i]
            query_time = query_times[i] if query_times.dim() > 0 else query_times
            time_window = time_windows[i] if time_windows.dim() > 0 else time_windows
            

            
            # Multi-hop expansion with early termination
            current_entities = {entity_id.item()}
            subgraph_edges = []
            edge_weights = []
            max_edges_per_hop = self.max_context_size // max(1, self.max_hops)  # Distribute edges across hops
            
            for hop in range(self.max_hops):
                next_entities = set()
                hop_edge_count = 0
                hop_decay = (0.8 ** hop)
                
                # Process current entities in batch when possible
                current_entities_list = list(current_entities)
                
                for ent in current_entities_list:
                    # Early termination if we have enough edges for this hop
                    if hop_edge_count >= max_edges_per_hop:
                        break
                        
                    ent_tensor = torch.tensor([ent], device=device, dtype=torch.long)
                    neighbors, weights = self._extract_temporal_neighbors(ent_tensor[0], query_time, time_window)
                    
                    if neighbors.size(0) > 0:
                        # Limit neighbors per entity to control expansion
                        max_neighbors = min(neighbors.size(0), max_edges_per_hop - hop_edge_count)
                        
                        # Take top-k neighbors by weight if we need to limit
                        if max_neighbors < neighbors.size(0):
                            _, top_indices = torch.topk(weights, max_neighbors)
                            neighbors = neighbors[top_indices]
                            weights = weights[top_indices]
                        
                        # Vectorized edge creation
                        for j in range(neighbors.size(0)):
                            neighbor_ent = neighbors[j, 0]  # Entity ID
                            rel = neighbors[j, 1]           # Relation ID
                            time = neighbors[j, 2]          # Time ID
                            subgraph_edges.append([ent, neighbor_ent.item(), rel.item(), time.item()])
                            edge_weights.append(weights[j].item() * hop_decay)
                            next_entities.add(neighbor_ent.item())
                            hop_edge_count += 1
                
                # Early termination conditions
                if not next_entities or len(subgraph_edges) >= self.max_context_size:
                    break
                    
                current_entities.update(next_entities)
            
            # Convert to tensors efficiently
            if subgraph_edges:
                subgraph_tensor = torch.tensor(subgraph_edges, device=device, dtype=torch.long)
                weight_tensor = torch.tensor(edge_weights, device=device, dtype=torch.float)
            else:
                subgraph_tensor = torch.empty((0, 4), device=device, dtype=torch.long)
                weight_tensor = torch.empty((0,), device=device, dtype=torch.float)
            
            all_subgraph_data.append(subgraph_tensor)
            all_temporal_weights.append(weight_tensor)
        
        return all_subgraph_data, all_temporal_weights

    def create_dynamic_temporal_subgraphs(self, ids, query_times, t_gnn=None):
        """Create dynamic temporal subgraphs with adaptive time windows - optimized"""
        device = self.device  # Use the model's device consistently
        n = ids.size(0)
        
        # Ensure inputs are on the correct device
        ids = ids.to(device)
        query_times = query_times.to(device)
        
        # Batch embedding computation for efficiency
        with torch.no_grad():  # No gradients needed for embeddings during subgraph creation
            entity_embs = self._entity_embedder().embed(ids).to(device)
            time_embs = self._time_embedder().embed(query_times).to(device)
        
        # Compute adaptive time windows
        time_windows = self._compute_adaptive_time_window(entity_embs, time_embs, query_times).to(device)
        
        # Extract temporal subgraphs
        subgraph_data_list, temporal_weights_list = self._expand_temporal_subgraph(ids, query_times, time_windows)
        
        g_list = []
        local_s_list = []
        num_nodes_list = []
        num_relations_total = self.dataset.num_relations()
        
        # Pre-allocate lists for better performance
        g_list = []
        local_s_list = []
        num_nodes_list = []
        
        # Batch process entity embeddings for minimal graphs
        minimal_graph_indices = []
        regular_graph_indices = []
        
        for i in range(n):
            if subgraph_data_list[i].size(0) == 0:
                minimal_graph_indices.append(i)
            else:
                regular_graph_indices.append(i)
        
        # Process minimal graphs in batch
        if minimal_graph_indices:
            minimal_entities = ids[minimal_graph_indices]
            minimal_embs = self._entity_embedder().embed(minimal_entities)
            
            for idx, i in enumerate(minimal_graph_indices):
                # Create minimal graph with just the seed entity
                g_i = dgl.graph((torch.tensor([0], device=device),
                               torch.tensor([0], device=device)),
                               num_nodes=1, idtype=torch.long).to(device)
                
                e = minimal_embs[idx:idx+1]
                if t_gnn is not None:
                    t_i = t_gnn[i:i+1]
                    e_diffused, _ = self._add_noise(e, t_i)
                    g_i.ndata['feat'] = e_diffused
                else:
                    g_i.ndata['feat'] = e
                
                g_i.edata['etype'] = torch.tensor([0], device=device)
                g_i.edata['temporal_weight'] = torch.tensor([1.0], device=device)
                
                g_list.append(g_i)
                local_s_list.append(0)
                num_nodes_list.append(1)
        
        # Process regular graphs
        for i in regular_graph_indices:
            s_i = ids[i]
            subgraph_edges = subgraph_data_list[i]
            edge_weights = temporal_weights_list[i]
            
            # Extract unique entities
            src_entities = subgraph_edges[:, 0]
            dst_entities = subgraph_edges[:, 1]
            all_entities = torch.unique(torch.cat([src_entities, dst_entities]))
            
            # Ensure seed entity is included
            if s_i.item() not in all_entities.tolist():
                all_entities = torch.cat([torch.tensor([s_i], device=device), all_entities])
            
            # Create entity mapping
            global_to_local = {ent.item(): idx for idx, ent in enumerate(all_entities)}
            local_s_i = global_to_local[s_i.item()]
            
            # Build edge lists with temporal weights
            edges_src = []
            edges_dst = []
            edges_type = []
            edge_temporal_weights = []
            
            for j, (src, dst, rel, time) in enumerate(subgraph_edges):
                if src.item() in global_to_local and dst.item() in global_to_local:
                    local_src = global_to_local[src.item()]
                    local_dst = global_to_local[dst.item()]
                    
                    # Forward edge
                    edges_src.append(local_src)
                    edges_dst.append(local_dst)
                    edges_type.append(rel.item())
                    edge_temporal_weights.append(edge_weights[j].item())
                    
                    # Inverse edge
                    edges_src.append(local_dst)
                    edges_dst.append(local_src)
                    edges_type.append(rel.item() + num_relations_total)
                    edge_temporal_weights.append(edge_weights[j].item())
            
            # Create DGL graph
            if edges_src:
                g_i = dgl.graph((torch.tensor(edges_src, device=device),
                               torch.tensor(edges_dst, device=device)),
                               num_nodes=len(all_entities), idtype=torch.long).to(device)
                
                # Add temporal weights as edge features
                g_i.edata['temporal_weight'] = torch.tensor(edge_temporal_weights, device=device)
            else:
                # Fallback: create self-loop graph
                g_i = dgl.graph((torch.tensor([local_s_i], device=device),
                               torch.tensor([local_s_i], device=device)),
                               num_nodes=len(all_entities), idtype=torch.long).to(device)
                g_i.edata['temporal_weight'] = torch.tensor([1.0], device=device)
                edges_type = [0]  # dummy relation
            
            # Set node features
            e = self._entity_embedder().embed(all_entities).to(device)
            if t_gnn is not None:
                t_i = t_gnn[i].expand(e.size(0))
                e_diffused, _ = self._add_noise(e, t_i)
                g_i.ndata['feat'] = e_diffused
            else:
                g_i.ndata['feat'] = e
            
            g_i.edata['etype'] = torch.tensor(edges_type, device=device)
            
            g_list.append(g_i)
            local_s_list.append(local_s_i)
            num_nodes_list.append(len(all_entities))
        
        # Batch graphs
        g_batched = dgl.batch(g_list)
        
        # Compute global indices for seed entities
        offset = 0
        global_s_idx_list = []
        for i in range(n):
            global_s_idx = offset + local_s_list[i]
            global_s_idx_list.append(global_s_idx)
            offset += num_nodes_list[i]
        
        return g_batched, torch.tensor(global_s_idx_list, device=device)

    def create_subgraphs(self, ids, ctx_list, ctx_size, t_gnn=None, query_times=None):
        """Wrapper method that chooses between static and dynamic subgraph creation"""
        if query_times is not None and hasattr(self, 'adaptive_window') and self.adaptive_window:
            return self.create_dynamic_temporal_subgraphs(ids, query_times, t_gnn)
        else:
            # Fallback to original static method
            return self.create_static_subgraphs(ids, ctx_list, ctx_size, t_gnn)
    
    def create_static_subgraphs(self, ids, ctx_list, ctx_size, t_gnn=None):
        """Original static subgraph creation method"""
        device = ids.device
        n = ids.size(0)
        g_list = []
        local_s_list = []
        num_nodes_list = []
        num_relations_total = self.dataset.num_relations()
        for i in range(n):
            s_i = ids[i]
            num_neighbors_i = ctx_size[s_i]
            neighbor_data_i = ctx_list[s_i, :num_neighbors_i, :]  # [num_neighbors_i, 3]
            neighbor_entities_i = neighbor_data_i[:, 0].long()
            relations_i = neighbor_data_i[:, 1].long()

            # build mapping
            all_nodes_i = torch.unique(torch.cat([torch.tensor([s_i], device=device), neighbor_entities_i]))
            global_to_local = {node.item(): idx for idx, node in enumerate(all_nodes_i)}
            local_s_i = global_to_local[s_i.item()]

            # edges lists
            edges_src = []
            edges_dst = []
            edges_type = []

            # 1-hop edges + inverse edges
            for ne, rel in zip(neighbor_entities_i, relations_i):
                local_ne = global_to_local[ne.item()]
                edges_src.append(local_s_i)
                edges_dst.append(local_ne)
                edges_type.append(rel.item())

                # inverse edge
                edges_src.append(local_ne)
                edges_dst.append(local_s_i)
                edges_type.append(rel.item() + num_relations_total)

            # (optional) self-loop omitted as RelGraphConv already adds it via self_loop=True

            g_i = dgl.graph((torch.tensor(edges_src, device=device),
                             torch.tensor(edges_dst, device=device)),
                            num_nodes=len(all_nodes_i), idtype=torch.long).to(device)

            e = self._entity_embedder().embed(all_nodes_i).to(device)
            if t_gnn is not None:
                t_i = t_gnn[i].expand(e.size(0))
                e_diffused, _ = self._add_noise(e, t_i)
                g_i.ndata['feat'] = e_diffused
            else:
                g_i.ndata['feat'] = e
            g_i.edata['etype'] = torch.tensor(edges_type, device=device)
            g_list.append(g_i)
            local_s_list.append(local_s_i)
            num_nodes_list.append(len(all_nodes_i))
        g_batched = dgl.batch(g_list)
        offset = 0
        global_s_idx_list = []
        for i in range(n):
            global_s_idx = offset + local_s_list[i]
            global_s_idx_list.append(global_s_idx)
            offset += num_nodes_list[i]
        return g_batched, torch.tensor(global_s_idx_list, device=device)

    def _get_encoder_output(self, p_emb, t_emb, ids, gt_ent, gt_rel, gt_tim, t_ids, output_repr=False):
        n = p_emb.size(0)
        device = p_emb.device
        if self.training:
            t_gnn = torch.randint(0, self.diffusion_steps, (n,), device=device)
        else:
            t_gnn = torch.zeros(n, device=device, dtype=torch.long)
        
        # Use dynamic temporal subgraph extraction if t_ids is available
        if t_ids is not None and hasattr(self, 'adaptive_window') and self.adaptive_window:
            # Ensure all inputs are on the same device
            ids = ids.to(device)
            t_ids = t_ids.to(device)
            g_batched, global_s_idx_list = self.create_subgraphs(ids, None, None, t_gnn=t_gnn, query_times=t_ids)
        else:
            # Fallback to static method
            ctx_list, ctx_size = self.dataset.index('neighbor')
            ctx_list = ctx_list.to(ids.device)
            ctx_size = ctx_size.to(ids.device)
            g_batched, global_s_idx_list = self.create_subgraphs(ids, ctx_list, ctx_size, t_gnn=t_gnn)
        
        out = self.gnn(g_batched)
        e_emb = out[global_s_idx_list]
        x = torch.cat([e_emb, p_emb, t_emb], dim=1)

        # --- diffusion regularisation (only on e_emb) ---
        if self.training:
            t_diff = torch.randint(0, self.diffusion_steps, (n,), device=device)
            e_t, epsilon = self._add_noise(e_emb, t_diff)
            cond_vec = torch.cat([p_emb, t_emb], dim=1)
            epsilon_pred = self._denoise(e_t, t_diff, cond_vec)
            diffusion_loss = F.mse_loss(epsilon_pred, epsilon)
        else:
            diffusion_loss = 0

        ctx_list, ctx_size = self.dataset.index('neighbor')
        ctx_list = ctx_list.to(device)
        ctx_size = ctx_size.to(device)
        ctx_ids = ctx_list[ids].to(device).transpose(1, 2)
        ctx_size = ctx_size[ids].to(device)

        if self.training:
            perm_vector = sc.get_randperm_from_lengths(ctx_size, ctx_ids.size(1))
            ctx_ids = torch.gather(ctx_ids, 1, perm_vector.unsqueeze(-1).expand_as(ctx_ids))

        ctx_ids = ctx_ids[:, :self.max_context_size]
        ctx_size[ctx_size > self.max_context_size] = self.max_context_size

        entity_ids = ctx_ids[..., 0]
        relation_ids = ctx_ids[..., 1]
        time_ids = ctx_ids[..., 2]

        ctx_size = ctx_size + 2
        attention_mask = sc.get_mask_from_sequence_lengths(ctx_size, self.max_context_size + 2)
        mask_neighbor = attention_mask[:, 2:].clone()

        if self.training and not output_repr:
            gt_mask = ((entity_ids != gt_ent.view(n, 1)) | (relation_ids != gt_rel.view(n, 1))) | (
                    time_ids != gt_tim.view(n, 1))
            ctx_random_mask = (attention_mask
                               .new_ones((n, self.max_context_size))
                               .bernoulli_(1 - self.get_option("ctx_dropout")))
            attention_mask[:, 2:] = attention_mask[:, 2:] & ctx_random_mask & gt_mask

        entity_emb = self._entity_embedder().embed(entity_ids)
        relation_emb = self._relation_embedder().embed(relation_ids)
        time_emb = self._time_embedder().embed(time_ids)

        # if self.training:
        #     x_neighbor = torch.cat([entity_emb.unsqueeze(2), relation_emb.unsqueeze(2), time_emb.unsqueeze(2)],
        #                            dim=2).view(n, self.max_context_size, -1)
        #     x_neighbor_flat = x_neighbor.view(-1, x_neighbor.size(2))
        #     t_diff_neighbor = torch.randint(0, self.diffusion_steps, (n * self.max_context_size,), device=device)
        #     x_t_neighbor_flat, epsilon_neighbor_flat = self._add_noise(x_neighbor_flat, t_diff_neighbor)
        #     epsilon_pred_neighbor_flat = self._denoise(x_t_neighbor_flat, t_diff_neighbor)
        #     mask_flat = mask_neighbor.view(-1)
        #     if mask_flat.sum() > 0:
        #         diffusion_loss_neighbor = F.mse_loss(epsilon_pred_neighbor_flat[mask_flat],
        #                                              epsilon_neighbor_flat[mask_flat])
        #     else:
        #         diffusion_loss_neighbor = 0
        #     diffusion_loss_total = diffusion_loss + diffusion_loss_neighbor
        # else:
        #     diffusion_loss_total = 0

        if self.training and self.get_option("self_dropout") > 0 and self.max_context_size > 0 and not output_repr and self.get_option("add_mlm_loss"):
            self_dropout_sample = sc.get_bernoulli_mask([n], self.get_option("self_dropout"), device)
            masked_sample = sc.get_bernoulli_mask([n], self.get_option("mlm_mask"), device) & self_dropout_sample
            t_emb[masked_sample] = self.local_mask.unsqueeze(0)
            replaced_sample = sc.get_bernoulli_mask([n], self.get_option("mlm_replace"), device) & self_dropout_sample & ~masked_sample
            t_emb[replaced_sample] = self._time_embedder().embed(torch.randint(self.dataset.num_times(),
                                                                               replaced_sample.shape, dtype=torch.long, device=device))[replaced_sample].detach()

        src = torch.cat(
            [torch.stack([e_emb, p_emb, t_emb], dim=1), torch.stack([entity_emb, relation_emb, time_emb], dim=2)
            .view(n, 3 * self.max_context_size, self.dim)], dim=1)
        src = src.reshape(n, self.max_context_size + 1, 3, self.dim)
        src = src[attention_mask[:, 1:]]
        pos = self.atomic_type_embeds(torch.arange(0, 4, device=device)).unsqueeze(0).repeat(src.shape[0], 1, 1)
        src = torch.cat([self.cls.expand(src.size(0), 1, self.dim), src], dim=1) + pos
        src = F.dropout(src, p=self.get_option("output_dropout"), training=self.training and not output_repr)
        src = self.atomic_layer_norm(src)
        out = self.atom_encoder(src,
                                self.convert_mask(src.new_ones(src.size(0), src.size(1), dtype=torch.long)),
                                output_all_encoded_layers=False)[-1][:, 0]
        src = out.new_zeros(n, self.max_context_size + 1, self.dim)
        src[attention_mask[:, 1:]] = out
        if self.max_context_size == 0:
            return src[:, 0], 0
        src = F.dropout(src, p=self.get_option("hidden_dropout"), training=self.training)
        src = self.layer_norm(src)
        out = self.mixer_encoder(src)
        out_pooling = self.glb_avg_pooling(out.view(n, self.dim, -1)).view(n, self.dim)


        if self.training and self.get_option("add_mlm_loss") and self.get_option("self_dropout") > 0.0 and self_dropout_sample.sum() > 0:
            all_time_emb = self._time_embedder().embed_all()
            all_time_emb = F.dropout(all_time_emb, p=self.get_option("output_dropout"), training=self.training)
            source_scores = self.similarity(out_pooling, all_time_emb, False).view(n, -1)
            self_pred_loss = F.cross_entropy(
                source_scores[self_dropout_sample], t_ids[self_dropout_sample], reduction='mean')
            return out_pooling, e_emb, 2 * self_pred_loss + diffusion_loss
        else:
            return out_pooling, e_emb, 0

    def convert_mask_rat(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
        return attention_mask

    def convert_mask(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask.float()) * -10000.0
        return attention_mask

    def _scoring(self, s_emb, p_emb, o_emb, t_emb, is_pairwise, ids, gt_ent, gt_rel, gt_tim, t_ids):
        encoder_output, e_emb, self_pred_loss = self._get_encoder_output(p_emb, t_emb, ids, gt_ent, gt_rel, gt_tim, t_ids)
        o_emb = F.dropout(o_emb, p=self.get_option("output_dropout"), training=self.training)
        g_scores = self.similarity(e_emb, o_emb, is_pairwise).view(p_emb.size(0), -1)
        target_scores = self.similarity(encoder_output, o_emb, is_pairwise).view(p_emb.size(0), -1)
        if self.training:
            return target_scores, g_scores, self_pred_loss
        else:
            return target_scores, g_scores

    def score_emb(self, s_emb, p_emb, o_emb, t_emb, combine: str, s, o, gt_ent=None, gt_rel=None, gt_tim=None, t_ids=None):
        if combine == 'spoo' or combine == 'sp_' or combine == 'spo':
            out = self._scoring(s_emb, p_emb, o_emb, t_emb, combine.startswith('spo'), s, gt_ent, gt_rel, gt_tim, t_ids)
        elif combine == 'spos' or combine == '_po':
            out = self._scoring(o_emb, p_emb, s_emb, t_emb, combine.startswith('spo'), o, gt_ent, gt_rel, gt_tim, t_ids)
        else:
            raise Exception("Combine {} is not supported in SURGE's score function".format(combine))
        return out

class SURGE(KgeModel):
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, SURGEScorer, configuration_key=configuration_key)
        self.loss = KgeLoss.create(config)

    def forward(self, fn_name, *args, **kwargs):
        self._scorer._entity_embedder = self.get_s_embedder
        self._scorer._relation_embedder = self.get_p_embedder
        self._scorer._time_embedder = self.get_t_embedder
        scores = getattr(self, fn_name)(*args, **kwargs)
        del self._scorer._entity_embedder
        del self._scorer._relation_embedder
        del self._scorer._time_embedder
        if fn_name == 'get_hitter_repr':
            return scores
        if self.training:
            self_loss_w = self.get_option("self_dropout")
            self_loss_w = self_loss_w / (1 + self_loss_w)
            return self.loss(scores[0], kwargs["gt_ent"]) + self.loss(scores[1], kwargs["gt_ent"]) + self_loss_w * scores[2] * scores[0].size(0)
        else:
            return scores

    def get_hitter_repr(self, s, p):
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        return self._scorer._get_encoder_output(p_emb, s, None, None, output_repr=True)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, t: Tensor, direction=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        t_emb = self.get_t_embedder().embed(t)
        if direction:
            if direction == 's':
                p_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())
            else:
                p_emb = self.get_p_embedder().embed(p)
            return self._scorer.score_emb(s_emb, p_emb, o_emb, t_emb, "spo" + direction, s, o)[0].view(-1)
        else:
            raise Exception("The SURGE model cannot compute undirected spo scores.")

    def score_sp(self, s: Tensor, p: Tensor, t: Tensor, o: Tensor = None, gt_ent=None, gt_rel=None, gt_tim=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        t_emb = self.get_t_embedder().embed(t)
        if o is None:
            o_emb = self.get_o_embedder().embed_all()
        else:
            o_emb = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(s_emb, p_emb, o_emb, t_emb, "sp_", s, None, gt_ent, gt_rel, gt_tim, t)

    def score_po(self, p: Tensor, o: Tensor, t: Tensor, s: Tensor = None, gt_ent=None, gt_rel=None, gt_tim=None) -> Tensor:
        if s is None:
            s_emb = self.get_s_embedder().embed_all()
        else:
            s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        t_emb = self.get_t_embedder().embed(t)
        p_inv_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())
        return self._scorer.score_emb(s_emb, p_inv_emb, o_emb, t_emb, "_po", None, o, gt_ent, gt_rel, gt_tim, t)

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, t: Tensor, entity_subset: Tensor = None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        p_inv_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())
        o_emb = self.get_o_embedder().embed(o)
        t_emb = self.get_t_embedder().embed(t)
        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s_emb, p_emb, all_entities, t_emb, "sp_", s, None)[0]
            po_scores = self._scorer.score_emb(all_entities, p_inv_emb, o_emb, t_emb, "_po", None, o)[0]
        else:
            assert False
        return torch.cat((sp_scores, po_scores), dim=1)