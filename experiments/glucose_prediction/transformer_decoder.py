import math
import torch
import torch.nn as nn


class MultiBranchAutoregressiveDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.feature_n = len(args.obs_features)
        self.window_length = args.obs_window

        nhead = args.nhead
        T_context = args.input_window
        T_future = args.t_future
        num_decoder_layers = args.num_decoder_layers
        dropout = args.dropout
        
        self.Tc = T_context
        self.Tf = T_future
        self.TT = self.Tc + self.Tf
        self.d_model = d_model = self.feature_n #CHECK THIS
        

        assert args.obs_features[0] == 'cgm' #assumes first feature is the one to predict, which should be cgm

        # positional encodings for context+future
        self.pos_emb = nn.Parameter(torch.zeros(1, T_context+T_future, d_model, device=self.device))
        nn.init.trunc_normal_(self.pos_emb, std=0.02) #FIXME paramaterise

        # GPT‐style decoder layers
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True, device=self.device
        )
        self.decoder = nn.TransformerDecoder(layer, num_decoder_layers)

        # final projection to scalar
        self.output_proj = nn.Linear(d_model, 1, device=self.device) #consider if this should be removed

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _causal_mask(self, L, device):
        # upper triangular -inf mask, so the transformer can't learn off future values
        m = torch.triu(torch.full((L, L), float('-inf'), device=self.device), diagonal=1)
        return m.to(device)

    def forward(self, ctx, tgt_future=None, generate_features_func=None, training=True):
        """
        ctx: (B, Tc, D) feature stream
        tgt_future: (B, Tf, D) ground truth future if training, else None
        generate_features_func: function to generate secondary features, if any and not given by future
        returns: y_pred (B, Tf, 1)
        """
        B, Tc, D = ctx.shape
        assert Tc == self.Tc and D == self.d_model

        # 1) prepare buffer for decoder inputs
        # reserve space for context + either ground truth or predictions
        L = Tc + self.Tf

        # copy context into the first Tc slots
        dec_input = torch.zeros(B, L, D, device=self.device, dtype=ctx.dtype) # (B, L, D)
        dec_input[:, :Tc, :] = ctx

        # if training & tgt_future provided => teacher forcing
        use_teacher = (tgt_future is not None) and training

        # 2) autoregressive loop
        for t in range(self.Tf):
            # positional add
            inp = dec_input[:, :Tc + t, :] + self.pos_emb[:, :Tc + t, :]

            # causal mask for current length
            mask = self._causal_mask(Tc + t, self.device)

            # pass through decoder (no external memory)
            # memory can be zeros
            memory = torch.zeros(B, 0, D, device=self.device, dtype=dec_input.dtype)
            out = self.decoder(tgt=inp, memory=memory, tgt_mask=mask)
            # out: (B, Tc+t, D)

            # project last time-step
            next_token = self.output_proj(out[:, -1, :])   # (B, 1)

            if use_teacher:
                # overwrite with ground-truth, including features
                
                dec_input[:, Tc + t, :] = tgt_future[:, t, :].squeeze(-1)

                # note: you may want another linear to map scalar→D if dims mismatch
            else:

                if self.feature_n > 1 and generate_features_func != None: #override secondary features with generator function
                    feat = generate_features_func(dec_input[:, :Tc + t, :]) # (B, Tc+t, D) -> (B, D)
                    assert feat[:, 0] == dec_input[:, Tc+t, 0] #assert that feature doesn't reassign primary feature
                    dec_input[:, Tc + t, :] = feat.squeeze(-1).unsqueeze(-1)
                else:
                    dec_input[:, Tc + t, :] = tgt_future[:, t, :].squeeze(-1) #overwrite features

                    # use model prediction: project scalar back into D with a small proj
                    # here we reuse output_proj's weight transpose as a quick hack:
                    # D_out = output_proj(out) maps D→1, so use its `.weight.T` for 1→D

                    # next_d = next_token @ self.output_proj.weight      # (B, D)
                    # dec_input[:, Tc + t, :] = next_d

                    dec_input[:, Tc + t, 0] = next_token.squeeze(1) #assign next glucose value



        # 3) collect the final Tf predictions
        y_all = dec_input[:, Tc:, :]                         # (B, Tf, D)
        # if you kept them as scalars, instead track them in a buffer
        # here we assume last dimension is D but only the first dim matters:
        y_pred = self.output_proj(y_all)                     # (B, Tf, 1)
        return y_pred

    def forward_single(self, ctx, tgt_future, generate_features_func=None, training=True):
        """
        Converts from single stream into batching func.
        ctx: (Tc, D) feature stream
        tgt_future: (Tf, D) ground truth future if training, else None
        generate_features_func: function to generate secondary features, if any and not given by future
        returns: y_pred (Tf, 1)
        """
        batched_y_pred = self.forward(ctx.unsqueeze(0),  tgt_future.unsqueeze(0), lambda i : generate_features_func(i.unsqueeze(0)), training)
        return batched_y_pred[0, :, :]

    def update(self, ctx, tgt_future, y_map=None, loss_map=None):
        """Performs a single training cycle of the model, returning log info
        Args:
            ctx: (B, Tc, D) feature stream
            tgt_future: (B, Tf, D) ground truth future if training, else None
        """
        logs = dict()

        y_pred = self.forward(ctx, tgt_future, None, True).squeeze(-1) #(B, Tf)
        y_actual = tgt_future[:, :, 0] #(B, Tf)
        if y_map != None:
            y_pred.apply_(y_map)
            y_actual.apply_(y_map)
        loss = torch.sqrt(torch.mean((y_pred-y_actual)**2))


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        logs["loss"] = loss.detach().cpu().numpy()
        if loss_map != None: logs["loss"] = loss_map(logs["loss"])
        
        return logs
    
    def eval_update(self, ctx, tgt_future, y_map=None, loss_map=None):
        """Performs a single training cycle of the model, returning log info
        Args:
            ctx: (B, Tc, D) feature stream
            tgt_future: (B, Tf, D) ground truth future if training, else None
        """
        logs = dict()

        with torch.no_grad():
            y_pred = self.forward(ctx, tgt_future, None, False).squeeze(-1) #(B, Tf)
            y_actual = tgt_future[:, :, 0] #(B, Tf)

            if y_map != None:
                y_pred = y_map(y_pred)
                y_actual = y_map(y_actual)
            loss = torch.sqrt(torch.mean((y_pred-y_actual)**2))


        logs["loss"] = loss.detach().cpu().numpy()
        if loss_map != None: logs["loss"] = loss_map(logs["loss"])
        
        return logs


