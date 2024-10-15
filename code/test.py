import torch
outputs = torch.tensor([[0.1,0.2],
                        [0.3,0.4]])

# dim = 1按行最大
print(outputs.argmax(1))
preds = outputs.argmax(1)
targets = torch.tensor([0,1])
print((preds==targets).sum())