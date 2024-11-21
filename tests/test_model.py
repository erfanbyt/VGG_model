import unittest
import torch
import torch.nn as nn
import torchsummary as summary

from vgg_model.model import Block, VGG

class TestVGG(unittest.TestCase):
    def __init__(self):
        super().__init__()

    def test_CNN_block_output(self):
        block = Block(in_channels=3, out_channels=64, no_layers=2)
        input_tensor = torch.rand(1, 3, 224, 224)
        output_tensor = block(input_tensor)
        self.assertEqual(output_tensor, (1, 64, 112, 112))

    # def test_vgg_output_shape(self):
    #     vgg = VGG(model_type="VGG11")
    #     input_tensor = torch.rand(1, 3, 224, 224)
    #     output_tensor = vgg(input_tensor)
    #     self.assertEqual(output_tensor, (1, 1000))

    # def test_vgg_model_configs(self):
    #     # Test all VGG configurations
    #     configs = ["VGG11", "VGG13", "VGG16", "VGG19"]
    #     for config in configs:
    #         vgg = VGG(model_type=config)
    #         input_tensor = torch.rand(1, 3, 224, 224)
    #         output_tensor = vgg(input_tensor)
    #         self.assertEqual(output_tensor.shape, (1, 1000))

    # def test_vgg_invalid_config(self):
    #     with self.assertRaises(KeyError):
    #         VGG(model_type="INVALID")

    # def test_block_with_edge_case(self):
    #     block = Block(in_channels=3, out_channels=64, no_layers=2)
    #     input_tensor = torch.rand(1, 5, 224, 224)  # Mismatch in input channels
    #     with self.assertRaises(RuntimeError):
    #         block(input_tensor)

    # def test_vgg_summary(self):
    #     # Test if the summary of the model works as expected
    #     vgg = VGG(model_type="VGG11")
    #     try:
    #         summary(vgg, input_size=(3, 224, 224))
    #     except Exception as e:
    #         self.fail(f"Model summary failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()