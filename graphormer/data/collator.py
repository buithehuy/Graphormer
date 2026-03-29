# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_float_unsqueeze(x, padlen):
    """Float-safe version: does NOT add +1 offset (which would corrupt RGB/centroid features)."""
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_float_unsqueeze(x, padlen1, padlen2):
    # Dùng cho các trường hợp features như ảnh [C, H, W] mà kích thước chuẩn rồi, 
    # nhưng có thể cần giữ api tương tự (ở đây raw_image thường fix kích thước 3x224x224, 
    # chỉ cần unsqueeze)
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            getattr(item, "num_nodes", item.x.size(0)),  # real node count for float features
            getattr(item, "raw_image", None),            # [3, 224, 224] tensor if online CNN is enabled
            getattr(item, "pos", None),                  # [N, 2] tensor if online CNN is enabled
            getattr(item, "node_type", None),            # [N] long: 0=fine, 1=coarse (C1)
            getattr(item, "mask", None),                 # [224, 224] long tensor for mask pooling
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        num_nodes_list,
        raw_images,
        pos_list,
        node_type_list,
        masks_list,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    # Use float-safe padding for x (no +1 offset — features are float RGB+centroid in [0,1])
    is_float_x = xs[0].dtype == torch.float32 or xs[0].dtype == torch.float16
    if is_float_x:
        x = torch.cat([pad_2d_float_unsqueeze(i, max_node_num) for i in xs])
    else:
        x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    # Online CNN handling
    raw_images_batch = None
    if raw_images[0] is not None:
        raw_images_batch = torch.stack(list(raw_images), dim=0)
    
    pos_batch = None
    if pos_list[0] is not None:
        pos_batch = torch.cat([pad_2d_float_unsqueeze(i, max_node_num) for i in pos_list])

    # C1: Hierarchical node type: 0=fine, 1=coarse. None for legacy graphs.
    node_type_batch = None
    if node_type_list[0] is not None:
        # Pad to max_node_num with 0 (fine) — padding nodes are masked out anyway
        node_type_batch = torch.cat([
            torch.cat([
                nt,
                torch.zeros(max_node_num - nt.size(0), dtype=torch.long)
            ]).unsqueeze(0)
            for nt in node_type_list
        ])  # [B, max_node_num]

    mask_batch = None
    if masks_list[0] is not None:
        mask_batch = torch.stack(list(masks_list), dim=0) # [B, 224, 224]

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,
        num_nodes=torch.tensor(num_nodes_list, dtype=torch.long),  # real node counts
        raw_image=raw_images_batch,
        pos=pos_batch,
        node_type=node_type_batch,  # C1: [B, N] or None
        mask=mask_batch,
    )
