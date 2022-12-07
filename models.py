import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


def get_pretrained_model(model_name):
    res_model = torch.hub.load(
        "pytorch/vision:v0.10.0",
        model_name,
        weights="ResNet152_Weights.DEFAULT",
    )
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    res_model.eval()
    # remove the last layer
    # new_model = torch.nn.Sequential(*(list(res_model.children())[:-1]))
    # free the model
    for param in res_model.parameters():
        param.requires_grad = False

    return res_model


class NeuralNet(nn.Module):
    def __init__(self, input_features=1000, n_classes=11):
        super(NeuralNet, self).__init__()
        # self.fc1 = nn.Linear(input_features, 256)
        self.out = nn.Linear(input_features, n_classes)

    def forward(self, x):
        # x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2)  # with 20% dropout rate
        x = self.out(x)
        return x


def get_custom_model(res_out_features, n_classes):
    """returns a resnet model with fully connected layers added to it:
    the top part is a layers with frozen resnet model's weights.
    the lower part is a neural network with 2 fully connected layers.
    """
    models = ["resnet50", "resnet101", "resnet152"]
    res_model = get_pretrained_model(models[2])
    nn_model = NeuralNet(res_out_features, classes)
    custom_model = nn.Sequential(res_model, nn_model)
    return custom_model


if __name__ == "__main__":
    models = ["resnet50", "resnet101", "resnet152"]
    res_out_features = 1000
    classes = 11
    custom_model = get_custom_model(res_out_features, classes)
    # res_model = get_pretrained_model(models[2])
    # nn_model = NeuralNet(res_out_features, classes)
    # custom_model = nn.Sequential(res_model, nn_model)

    print(summary(custom_model, (3, 224, 224)))
