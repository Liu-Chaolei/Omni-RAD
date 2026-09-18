"""CPU tests of conditioning behavior without SD-Turbo or entropy weights."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from rate_control import LambdaFiLMEmbed, FiLMLayer, DynamicTimestepModule


@unittest.skipIf(torch is None, 'PyTorch is required for conditioning tests')
class RateControlTests(unittest.TestCase):
    def test_extreme_scales_and_batch_independence(self):
        alpha = torch.linspace(0.999, 0.001, 1000)
        module = DynamicTimestepModule(alpha)
        scales = torch.tensor([0.0, 0.1, 100.0]).view(3, 1, 1, 1).expand(3, 8, 4, 4)
        result = module(scales)
        self.assertTrue(torch.all((result >= 800) & (result <= 999)))
        self.assertTrue(torch.all(result[:-1] >= result[1:]))
        self.assertEqual(result[0].item(), 999)
        self.assertEqual(result[-1].item(), 800)
        for i in range(3):
            torch.testing.assert_close(result[i:i+1], module(scales[i:i+1]))

    def test_film_identity_and_saved_bounds(self):
        embed = LambdaFiLMEmbed(lambda_min=0.2, lambda_max=128)
        conditions = embed(torch.tensor([0.2, 2.0, 128.0]))
        self.assertEqual(tuple(conditions.shape), (3, 512))
        features = torch.randn(3, 8, 4, 4)
        layer = FiLMLayer(512, 8)
        torch.testing.assert_close(layer(features, conditions), features)
        restored = LambdaFiLMEmbed(lambda_min=1, lambda_max=2)
        restored.load_state_dict(embed.state_dict())
        torch.testing.assert_close(restored(torch.tensor([0.2, 2, 128])), conditions)


if __name__ == '__main__':
    unittest.main()
