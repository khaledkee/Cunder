import torch
import torch.nn


class SimpleNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1: torch.nn.Linear = torch.nn.Linear(1, 3)
        self.fc2: torch.nn.Linear = torch.nn.Linear(1, 3)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return [self.fc1(x), self.fc2(y), torch.cat([self.fc1(x), self.fc2(x)], dim=0)]


if __name__ == "__main__":
    scripted_model = torch.jit.script(SimpleNet2())
    torch.jit.save(scripted_model, 'model_2_input_3_output.pt')
