import torch 

# class ParamNNLayer(torch.nn.Module):
#   def __init__(self, n_nodes_in, n_nodes_out, dropout):
#     super().__init__()
#     self.Linear = torch.nn.Linear(n_nodes_in, n_nodes_out-2)
#     self.Dropout = torch.nn.Dropout(dropout)
#     self.ELU = torch.nn.ELU()

#   def forward(self, x):
#     x1 = self.Linear(x)
#     x2 = self.Dropout(x1)
#     x3 = self.ELU(x2)

#     out = torch.cat([x3, x[:,-2:]], dim=1)
#     return out

class PassThroughLayer(torch.nn.Module):
  def __init__(self, n_nodes_in, n_nodes_out, dropout):
    super().__init__()
    self.Linear = torch.nn.Linear(n_nodes_in-2, n_nodes_out-2)
    self.Dropout = torch.nn.Dropout(dropout)
    self.ELU = torch.nn.ELU()

  def forward(self, x):
    x1 = self.Linear(x[:,:-2])
    x2 = self.Dropout(x1)
    x3 = self.ELU(x2)

    out = torch.cat([x3, x[:,-2:]], dim=1)
    return out


