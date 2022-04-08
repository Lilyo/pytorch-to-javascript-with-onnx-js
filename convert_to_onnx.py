import torch

from inference_mnist_model import Net, ConvMGU_ONNX


def main():
    # pytorch_model = Net()
    pytorch_model = ConvMGU_ONNX(num_classes=10, latent_dim=512, lstm_layers=1, hidden_dim=512)
    # pytorch_model.load_state_dict(torch.load('pytorch_model.pt'))
    pytorch_model.eval()

    h0 = torch.randn(512)
    dummy_input = torch.zeros(112 * 112 * 4)
    torch.onnx.export(pytorch_model, (dummy_input, h0), './full_demo/onnx_model.onnx', opset_version=11, verbose=True)


if __name__ == '__main__':
    main()
