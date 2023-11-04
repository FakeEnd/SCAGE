from models.model import GraphTransformer


def get_model(args):
    return GraphTransformer(
        use_sms=args['use_sms'],
        num_encoder_layers=args['num_encoder_layers'],
        hidden_dim=args['hidden_dim'],
        ffn_hidden_dim=args['ffn_hidden_dim'],
        num_attn_heads=args['num_attn_heads'],
        emb_dropout=args['emb_dropout'],
        dropout=args['dropout'],
        attn_dropout=args['attn_dropout'],
        dist_bar=args['dist_bar'],
        encoder_normalize_before=args['encoder_normalize_before'],
        apply_graphormer_init=args['apply_graphormer_init'],
        activation_fn=args['activation_fn'],
        n_trans_layers_to_freeze=args['n_trans_layers_to_freeze'],

        use_super_node=args['use_super_node'],
        graph_pooling=args['graph_pooling'],
        afps_k=args['afps_k'],

        node_level_modules=args['node_level_modules'],
        attn_mask_modules=args['attn_mask_modules'],

        num_in_degree=args['num_in_degree'],
        num_out_degree=args['num_out_degree'],
        eig_pos_dim=args['eig_pos_dim'],
        svd_pos_dim=args['svd_pos_dim'],

        use_gnn_layers=args['use_gnn_layers'],
        residual=args['residual'],
        num_gnn_layers=args['num_gnn_layers'],
        JK=args['JK'],
        gnn_dropout=args['gnn_dropout']
    )
