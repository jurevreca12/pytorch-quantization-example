from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from main import Net, get_data_loaders, test

class PTSQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.fp32_model = Net()
        self.fp32_model.load_state_dict(torch.load('mnist_cnn.pt'))
        self.fp32_model.eval()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fp32_model(x)
        x = self.dequant(x)
        return x


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST example of static post-training quantization.')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    model = PTSQNet().to(device)
    # model must be set to eval mode for static quantization logic to work
    model.eval()

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'x86' for server inference and 'qnnpack'
    # for mobile inference. Other quantization configurations such as selecting
    # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
    # can be specified here.
    # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
    # for server inference.
    # model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    model_fused = torch.ao.quantization.fuse_modules(model.fp32_model, [['conv1', 'relu1'],
                                                                        ['conv2', 'relu2'],
                                                                        ['fc2',   'relu3']])

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_prepared = torch.ao.quantization.prepare(model_fused)

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    input_fp32 = torch.randn(1, 1, 28, 28)
    model_prepared(input_fp32)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.ao.quantization.convert(model_prepared)

    # run the model, relevant calculations will happen in int8
    model_int8(input_fp32)

    print("TESTING QUANTIZED MODEL")
    test_kwargs = {'batch_size': args.test_batch_size}
    _, test_loader = get_data_loaders(test_kwargs=test_kwargs)
    test(model_int8, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn_ptq_i8.pt")


if __name__ == '__main__':
    main()

