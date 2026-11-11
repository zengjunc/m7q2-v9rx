# 复现新增：验证短窗口的有限未来依赖、异常检测、SE(2)运动投影及记录器不变性。
import unittest
import numpy as np
from record_pf_candidates import make_recording_filter
from window_refine import analyze, odometry_steps, posterior_weights, refine, smooth_window


def toy_record(frames=31):
    rng = np.random.default_rng(13)
    stamps = np.arange(frames)*.1
    base = np.column_stack([stamps, np.zeros((frames, 2))])
    cloud = rng.normal(0, .04, (frames, 128, 3)) + base[:, None]
    return dict(poses=cloud, weights=np.full((frames, 128), 1/128),
                pf_pose=base, timestamps=stamps,
                initial_pose=np.array([-.1, 0, 0]),
                actions=np.tile([.1, 0, 0], (frames, 1)))


class WindowRefineTests(unittest.TestCase):
    def test_body_forward_rotates_into_map_north(self):
        r = toy_record(3)
        r['initial_pose'][2] = np.pi/2
        r['pf_pose'][:, 2] = np.pi/2
        np.testing.assert_allclose(odometry_steps(r), [[0, .1]]*3, atol=1e-12)

    def test_correct_odometry_is_unchanged_even_through_turn(self):
        yaw = np.linspace(0, 1., 31)
        r = toy_record()
        r['pf_pose'][:, 2] = yaw
        delta = odometry_steps(r)
        r['pf_pose'][:, :2] = r['initial_pose'][:2] + np.cumsum(delta, axis=0)
        result, _ = refine(r, trigger='all', lag=5)
        np.testing.assert_allclose(result, r['pf_pose'], atol=1e-10)

    def test_local_outlier_reduced_without_touching_heading(self):
        r = toy_record()
        r['pf_pose'][15, 1] = 1.
        result, gate = refine(r, trigger='jump', lag=5)
        self.assertLess(abs(result[15, 1]), 1.)
        self.assertGreater(gate[15], 0.)
        np.testing.assert_array_equal(result[:, 2], r['pf_pose'][:, 2])
        np.testing.assert_array_equal(result[:10], r['pf_pose'][:10])
        self.assertLessEqual(np.linalg.norm(result-r['pf_pose'], axis=1).max(), .5+1e-10)

    def test_no_future_beyond_lag_and_committed_prefix_is_stable(self):
        r = toy_record()
        r['pf_pose'][15, 1] = 1.
        full, _ = refine(r, trigger='all', lag=3)
        prefix = {k: (v[:20] if k != 'initial_pose' else v) for k, v in r.items()}
        early, _ = refine(prefix, trigger='all', lag=3)
        np.testing.assert_allclose(full[:17], early[:17], atol=1e-12)
        r['pf_pose'][20:, :2] += 100
        changed, _ = refine(r, trigger='all', lag=3)
        np.testing.assert_allclose(full[:17], changed[:17], atol=1e-12)

    def test_zero_lag_is_exact_original(self):
        r = toy_record()
        r['pf_pose'][15, 1] = 1.
        result, _ = refine(r, trigger='all', lag=0)
        np.testing.assert_array_equal(result, r['pf_pose'])

    def test_two_separated_modes_detect_empty_mean(self):
        r = toy_record(3)
        r['poses'][:, :64, 1] -= 3
        r['poses'][:, 64:, 1] += 3
        stats = analyze(r)
        self.assertTrue(stats['valley'].all())
        # Changing a diagnostic cloud alone must not change PF output.
        np.testing.assert_array_equal(r['pf_pose'][:, 1], 0)

    def test_single_gaussian_is_not_automatically_two_modes(self):
        r = toy_record()
        # A broad unimodal cloud must not trigger merely because its spread is large.
        r['poses'] = r['pf_pose'][:, None] + 100*(r['poses']-r['pf_pose'][:, None])
        self.assertFalse(analyze(r)['valley'].any())
        result, _ = refine(r)
        np.testing.assert_array_equal(result, r['pf_pose'])

    def test_local_mode_route_uses_future_and_prefers_connected_branch(self):
        from window_refine import mode_targets
        base = np.zeros((3, 2))
        stats = dict(valley=np.array([False, True, False]),
                     centers=np.array([[[0, 0], [0, 0]], [[0, 0], [3, 0]], [[0, 0], [0, 0]]]),
                     masses=np.array([[1., 0], [.4, .6], [1., 0]]), steps=np.zeros((3, 2)))
        target = mode_targets(base, stats, 0, 3, 1.)
        self.assertAlmostEqual(target[1, 0], 0.)

    def test_reconstruct_weights_with_resample_reset(self):
        r = dict(log_likelihood=np.log([[.2, .8], [.9, .1], [.4, .6]]),
                 resampled=np.array([False, True, False]))
        weights = posterior_weights(r)
        np.testing.assert_allclose(weights, [[.2, .8], [.18/.26, .08/.26], [.4, .6]])

    def test_smoother_invariant_to_utm_translation(self):
        m = np.array([[0., 0], [1., 1], [2., 0]])
        steps = np.tile([1., 0], (3, 1))
        expected = smooth_window(m, steps, np.ones(3), 1.)
        origin = np.array([500000., 4500000.])
        actual = smooth_window(m+origin, steps, np.ones(3), 1.)-origin
        np.testing.assert_allclose(expected, actual, atol=1e-8)

    def test_recorder_keeps_pf_rng_and_output_unchanged(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
        import torch
        from particle_filter import ParticleFilter

        class Observation:
            def __call__(self, particles, *args, results=None, **kwargs):
                values = np.array([-5 * (p.x-.5) ** 2 + 2*np.cos(p.theta) for p in particles])
                results['log_likelihood'] = values
                return values

        def run(cls):
            np.random.seed(123)
            pf = cls(torch.nn.Linear(1, 1), n_particles=16, n_eff_threshold=.8,
                     resample_noise_sigma=(.1, .1, .01), action_gain=(.1, .1, .1),
                     drift_gain=(1, 1, 1), aerial_image_resize=(1, 1),
                     var_ema_alpha=.2, var_temp=.3, var_gamma=2., likelihood_temp=1.)
            pf.observation_model = Observation()
            pf.initialize(np.array([0., 0, .1]), np.array([1., 1., .2]))
            poses, sampled = [], []
            for action in [[0., 0, 0], [.2, .1, .03], [.3, -.1, -.05]]:
                pf.predict(np.array(action))
                pf.update(None, None, {}, {})
                sampled.append(pf.resample())
                poses.append(pf.pose.copy())
            return np.array(poses), sampled, np.random.normal(size=5), pf

        original = run(ParticleFilter)
        recorded = run(make_recording_filter(ParticleFilter))
        np.testing.assert_array_equal(original[0], recorded[0])
        self.assertEqual(original[1], recorded[1])
        np.testing.assert_array_equal(original[2], recorded[2])
        self.assertEqual(len(recorded[3].trace['poses']), 3)


if __name__ == '__main__':
    unittest.main()
