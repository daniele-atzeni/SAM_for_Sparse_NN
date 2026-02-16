import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], num_classes: int):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        self.linear_relu_stack = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        logits = self.output_layer(logits)
        return logits