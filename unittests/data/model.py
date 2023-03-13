import torch
import torch.nn


class SampleNet(torch.nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    scripted_model = torch.jit.script(SampleNet())
    scripted_model.save('model.pt')