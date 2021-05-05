import torch
from torch import nn
import torch.nn.init as init
import math


class WSDModel(nn.Module):

    def __init__(self, V, Y, D=300, dropout_prob=0.2, use_padding=False, positional=False, causal=False):
        super(WSDModel, self).__init__()
        self.use_padding = use_padding
        self.positional = positional
        self.causal = causal
        self.D = D
        self.pad_id = 0
        self.E_v = nn.Embedding(V, D, padding_idx=self.pad_id)
        self.E_y = nn.Embedding(Y, D, padding_idx=self.pad_id)
        init.kaiming_uniform_(self.E_v.weight[1:], a=math.sqrt(5))
        init.kaiming_uniform_(self.E_y.weight[1:], a=math.sqrt(5))

        self.W_A = nn.Parameter(torch.Tensor(D, D))
        self.W_O = nn.Parameter(torch.Tensor(D, D))

        init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_O, a=math.sqrt(5))

        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm([self.D])
        self.inf_tensor = nn.Parameter(torch.tensor([float('-inf')]), requires_grad=False)


    def attention(self, X, Q, mask):
        """
        Computes the contextualized representation of query Q, given context X, using the attention model.

        :param X:
            Context matrix of shape [B, N, D]
        :param Q:
            Query matrix of shape [B, k, D], where k equals 1 (in single word attention) or N (self attention)
        :param mask:
            Boolean mask of size [B, N] indicating padding indices in the context X.

        :return:
            Contextualized query and attention matrix / vector
        """

        # weights: a = softmax(q * W^A * X^T)
        # outputs: x^c = a * X * W^O
        QW_A = torch.matmul(Q, self.W_A)
        XT = X.transpose(1, 2)
        logits = torch.bmm(QW_A, XT)

        if self.use_padding:
            logits = torch.where(mask.unsqueeze(1), logits, self.inf_tensor)


        N = X.shape[1]

        if Q.shape[1] > 1:
            D = torch.zeros(N, N)
            if self.positional:
                I = torch.Tensor(list(range(0, N))).unsqueeze(1).expand(-1, N)
                J = I.T
                D = (I - J).to(logits.device)

                # currently causal implemented in a way that relies on positional as well
                if self.causal:
                    D = torch.where(D >= 0, D, self.inf_tensor)
                D = -torch.abs(D)

            D = D.to(logits.device)
            A = self.softmax(logits + D)
        else:
            A = self.softmax(logits)

        AX = torch.bmm(A, X)
        Q_c = torch.matmul(AX, self.W_O)

        return Q_c, A.squeeze()

    def forward(self, M_s, v_q=None):
        """
        :param M_s:
            [B, N] dimensional matrix containing token integer ids
        :param v_q:
            [B] dimensional vector containing query word indices within the sentences represented by M_s.
            This argument is only passed in single word attention mode.

        :return: logits and attention tensors.
        """

        X = self.dropout_layer(self.E_v(M_s))   # [B, N, D]
        
        Q = None
        if v_q is not None:

            # v_q - [B], each sample is index of query word
            # need to transform to Q of size [B, 1, D]
            v_q_expanded = v_q.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.D)
            gathered = torch.gather(X, 1, v_q_expanded)
            Q = gathered
            
            # Look up the gather() and expand() methods in PyTorch.
        else:
            Q = X


        mask = M_s.ne(self.pad_id)
        Q_c, A = self.attention(X, Q, mask)
        H = self.layer_norm(Q_c + Q)
        
        E_y = self.dropout_layer(self.E_y.weight)
        y_logits = (H @ E_y.T).squeeze()
        return y_logits, A
