# 复现改进验证：检查网格内软权重、无效深度和双目矛盾缓存。
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dataset.stereo import load_depth_quality
from model.bev_mapper import LiftSplatBEVMapper


class StereoPoolingTest(unittest.TestCase):
    def setUp(self):
        self.mapper = LiftSplatBEVMapper(16, (2, 2))
        for parameter in self.mapper.feat_attn.parameters():
            parameter.data.zero_()
        self.mapper.compute_geometry = lambda depth, *args: (
            torch.ones(depth.numel(), dtype=torch.bool), torch.zeros(depth.numel(), dtype=torch.long))
        self.features = torch.arange(4, dtype=torch.float32).view(1, 1, 2, 2).expand(1, 16, 2, 2)
        self.depth = torch.ones(1, 2, 2)
        self.info = (torch.eye(3)[None], torch.eye(4)[None], torch.tensor([.3]))

    def test_identity_and_common_factor(self):
        baseline = self.mapper(self.features, self.depth, *self.info)
        for factor in [1., .25]:
            actual = self.mapper(self.features, self.depth, *self.info,
                                 depth_quality=torch.full_like(self.depth, factor))
            for a, b in zip(actual, baseline):
                self.assertTrue(torch.equal(a, b))

    def test_mixed_quality_weighted_mean(self):
        quality = torch.tensor([[[1., .25], [1., .25]]])
        result, _ = self.mapper(self.features, self.depth, *self.info, depth_quality=quality)
        self.assertAlmostEqual(result[0, 0, -1, -1].item(), (0 + .25 + 2 + .75) / 2.5)

    def test_invalid_and_empty_depth(self):
        self.depth[:, :, 1] = 0
        result, _ = self.mapper(self.features, self.depth, *self.info, depth_quality=torch.ones_like(self.depth))
        self.assertEqual(result[0, 0, -1, -1].item(), 1)
        self.mapper.mask_invalid_depth = True
        masked, _ = self.mapper(self.features, self.depth, *self.info)
        self.assertTrue(torch.equal(result, masked))
        result, mask = self.mapper(self.features, self.depth * 0, *self.info,
                                   depth_quality=torch.ones_like(self.depth))
        self.assertEqual(result.count_nonzero(), 0)
        self.assertEqual(mask.count_nonzero(), 0)

    def test_stereo_labels_and_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ['depth', 'left', 'right']:
                (root / name).mkdir()
            depth = np.zeros((2, 16), dtype=np.uint16)
            depth[0, 12:15] = 10015  # d≈10，分别落到右图x=2、3、4。
            cv2.imwrite(str(root / 'depth/000.png'), depth)
            for name in ['left', 'right']:
                cv2.imwrite(str(root / name / '000.png'), np.zeros_like(depth, dtype=np.uint8))
            right_disp = np.full(depth.shape, -128, dtype=np.int16)
            right_disp[0, 2] = -10  # 一致。
            right_disp[0, 3] = -8   # 矛盾。
            # x=4无右视差，未知应保留。
            matcher = SimpleNamespace(compute=lambda right, left: right_disp * 16)
            with patch('dataset.stereo.cv2.StereoSGBM_create', return_value=matcher):
                quality = load_depth_quality(root / 'depth/000.png', depth.shape)
            self.assertEqual(quality[0, 12].item(), 1)
            self.assertEqual(quality[0, 13].item(), .25)
            self.assertEqual(quality[0, 14].item(), 1)
            with patch('dataset.stereo.cv2.StereoSGBM_create', side_effect=AssertionError('Should reuse cache')):
                quality2 = load_depth_quality(root / 'depth/000.png', depth.shape, .5)
            self.assertEqual(quality2[0, 13].item(), .5)


if __name__ == '__main__':
    unittest.main()
