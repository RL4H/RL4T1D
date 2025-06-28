import math
import torch
import torch.nn as nn


class MultiBranchAutoregressiveDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.window_height = len(args.obs_features)
        self.window_length = args.obs_window

        nhead = args.nhead
        T_context = args.obs_window
        T_future = args.t_future
        num_decoder_layers = args.num_decoder_layers
        dropout = args.dropout
        
        self.Tc = T_context
        self.Tf = T_future
        d_model = self.window_height #CHECK THIS

        assert args.obs_features[0] == 'cgm' #assumes first feature is the one to predict, which should be cgm

        
        # fuse concat_dim=3*D → back to D
        self.fusion_proj = nn.Linear(3 * d_model, d_model)

        # positional encodings for context+future
        self.pos_emb = nn.Parameter(torch.zeros(1, T_context+T_future, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02) #FIXME paramaterise

        # GPT‐style decoder layers
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_decoder_layers)

        # final projection to scalar
        self.output_proj = nn.Linear(d_model, 1)

    def _causal_mask(self, L, device):
        # upper triangular -inf mask
        m = torch.triu(torch.full((L, L), float('-inf')), diagonal=1)
        return m.to(device)

    def forward(self, b1, b2, b3, tgt_future=None, generate_features_func=None):
        """
        b1,b2,b3: (B, Tc, D) feature streams
        tgt_future: (B, Tf, 1) ground truth future if training, else None
        returns: y_pred (B, Tf, 1)
        """
        B, Tc, D = b1.shape
        assert Tc == self.Tc and D == self.d_model

        # 1) fuse the three branches
        fused_ctx = torch.cat([b1, b2, b3], dim=2)      # (B, Tc, 3D)
        fused_ctx = self.fusion_proj(fused_ctx)         # (B, Tc, D)

        # 2) prepare buffer for decoder inputs
        # reserve space for context + either ground truth or predictions
        L = Tc + self.Tf
        device = fused_ctx.device

        # copy context into the first Tc slots
        dec_input = torch.zeros(B, L, D, device=device, dtype=fused_ctx.dtype)
        dec_input[:, :Tc, :] = fused_ctx

        # if training & tgt_future provided => teacher forcing
        use_teacher = (tgt_future is not None) and self.training

        # 3) autoregressive loop
        for t in range(self.Tf):
            # positional add
            inp = dec_input[:, :Tc + t, :] + self.pos_emb[:, :Tc + t, :]

            # causal mask for current length
            mask = self._causal_mask(Tc + t, device)

            # pass through decoder (no external memory)
            # memory can be zeros
            memory = torch.zeros(B, 0, D, device=device, dtype=dec_input.dtype)
            out = self.decoder(tgt=inp, memory=memory, tgt_mask=mask)
            # out: (B, Tc+t, D)

            # project last time-step
            next_token = self.output_proj(out[:, -1, :])   # (B, 1)

            if use_teacher:
                # overwrite with ground-truth
                dec_input[:, Tc + t, :] = tgt_future[:, t, :].squeeze(-1).unsqueeze(-1).expand(-1, -1, D)
                # note: you may want another linear to map scalar→D if dims mismatch
            else:
                # use model prediction: project scalar back into D with a small proj
                # here we reuse output_proj's weight transpose as a quick hack:
                # D_out = output_proj(out) maps D→1, so use its `.weight.T` for 1→D
                next_d = next_token @ self.output_proj.weight      # (B, D)
                dec_input[:, Tc + t, :] = next_d

        # 4) collect the final Tf predictions
        y_all = dec_input[:, Tc:, :]                         # (B, Tf, D)
        # if you kept them as scalars, instead track them in a buffer
        # here we assume last dimension is D but only the first dim matters:
        y_pred = self.output_proj(y_all)                     # (B, Tf, 1)
        return y_pred
