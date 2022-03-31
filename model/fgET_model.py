import math

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertAdam

class fgET(nn.Module):

    def __init__(self,
                 label_size,
                 elmo_dim,
                 repr_dropout=.2,
                 dist_dropout=.5,
                 latent_size=0,
                 svd=None,
                 ):
        super(fgET, self).__init__()
        self.label_size = label_size
        self.elmo_dim = elmo_dim

        self.attn_dim = 1
        self.attn_inner_dim = self.elmo_dim
        # Mention attention
        self.men_attn_linear_m = nn.Linear(self.elmo_dim, self.attn_inner_dim, bias=False)
        self.men_attn_linear_o = nn.Linear(self.attn_inner_dim, self.attn_dim, bias=False)
        # Context attention
        self.ctx_attn_linear_c = nn.Linear(self.elmo_dim, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_m = nn.Linear(self.elmo_dim, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_d = nn.Linear(1, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_o = nn.Linear(self.attn_inner_dim,
                                        self.attn_dim, bias=False)
        # Output linear layers
        self.repr_dropout = nn.Dropout(p=repr_dropout)
        self.output_linear = nn.Linear(self.elmo_dim * 2, label_size, bias=False)

        # SVD
        if svd:
            svd_mat = self.load_svd(svd)
            self.latent_size = svd_mat.size(1)
            self.latent_to_label.weight = nn.Parameter(svd_mat, requires_grad=True)
            self.latent_to_label.weight.requires_grad = False
        elif latent_size == 0:
            self.latent_size = int(math.sqrt(label_size))
        else:
            self.latent_size = latent_size
        self.latent_to_label = nn.Linear(self.latent_size, label_size,
                                         bias=False)
        self.latent_scalar = nn.Parameter(torch.FloatTensor([.1]))
        self.feat_to_latent = nn.Linear(self.elmo_dim * 2, self.latent_size,
                                        bias=False)
        # Loss function
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.mse = nn.MSELoss()
        # Relative position (distance)
        self.dist_dropout = nn.Dropout(p=dist_dropout)

    def load_svd(self, path):
        print('Loading SVD matrices')
        u_file = path + '-Ut'
        s_file = path + '-S'
        with open(s_file, 'r', encoding='utf-8') as r:
            s_num = int(r.readline().rstrip())
            mat_s = [[0] * s_num for _ in range(s_num)]
            for i in range(s_num):
                mat_s[i][i] = float(r.readline().rstrip())
        mat_s = torch.FloatTensor(mat_s)

        with open(u_file, 'r', encoding='utf-8') as r:
            mat_u = []
            r.readline()
            for line in r:
                mat_u.append([float(i) for i in line.rstrip().split()])
        mat_u = torch.FloatTensor(mat_u).transpose(0, 1)
        return torch.matmul(mat_u, mat_s) #.transpose(0, 1)

    def forward_nn(self, elmo_embeddings, men_mask, ctx_mask, dist, gathers):
        # Elmo contextualized embeddings

        men_attn = self.men_attn_linear_m(elmo_embeddings).tanh()
        men_attn = self.men_attn_linear_o(men_attn)
        men_attn = men_attn + (1.0 - men_mask.unsqueeze(-1)) * -10000.0
        men_attn = men_attn.softmax(1)
        men_repr = (elmo_embeddings * men_attn).sum(1)

        dist = self.dist_dropout(dist)
        ctx_attn = (self.ctx_attn_linear_c(elmo_embeddings) +
                    self.ctx_attn_linear_m(men_repr.unsqueeze(1)) +
                    self.ctx_attn_linear_d(dist.unsqueeze(2))).tanh()
        ctx_attn = self.ctx_attn_linear_o(ctx_attn)

        ctx_attn = ctx_attn + (1.0 - ctx_mask.unsqueeze(-1)) * -10000.0
        ctx_attn = ctx_attn.softmax(1)
        ctx_repr = (elmo_embeddings * ctx_attn).sum(1)

        # Classification
        final_repr = torch.cat([men_repr, ctx_repr], dim=1)
        final_repr = self.repr_dropout(final_repr)
        outputs = self.output_linear(final_repr)

        outputs_latent = None
        latent_label = self.feat_to_latent(final_repr) #.tanh()
        outputs_latent = self.latent_to_label(latent_label)
        outputs = outputs + self.latent_scalar * outputs_latent

        return outputs, outputs_latent

    def forward(self, elmo_embeddings, labels, men_mask, ctx_mask, dist, gathers):
        outputs, outputs_latent = self.forward_nn(elmo_embeddings, men_mask, ctx_mask, dist, gathers)
        loss = self.criterion(outputs, labels)
        return loss

    def configure_optimizers(self,weight_decay,lr,total_step):
        optimizer = BertAdam(filter(lambda x: x.requires_grad, self.parameters()),
                             lr=lr, warmup=.1,
                             weight_decay=weight_decay,
                             t_total=total_step)
        return optimizer

    def _prediction(self, outputs, predict_top=True):
        _, highest = outputs.max(dim=1)
        highest = highest.int().tolist()
        preds = (outputs.sigmoid() > .5).int()
        if predict_top:
            for i, h in enumerate(highest):
                preds[i][h] = 1
        return preds

    def predict(self, elmo_embeddings, men_mask, ctx_mask, dist, gathers, predict_top=True):
        self.eval()
        outputs, _ = self.forward_nn(elmo_embeddings, men_mask, ctx_mask, dist, gathers)
        scores = outputs.sigmoid()
        predictions = self._prediction(outputs, predict_top=predict_top)
        self.train()
        return predictions,scores