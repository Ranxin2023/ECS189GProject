import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.__version__)
print(torch.cuda.is_available())