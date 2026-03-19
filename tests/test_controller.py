"""Tests for teleopit.controllers.rl_policy — RLPolicyController.

ONNX model file is typically not available in test environments, so most tests
use skipif or mock the ONNX session. We test static helpers directly.
"""
from unittest.mock import MagicMock, patch
import importlib

import numpy as np
import pytest

from conftest import requires_onnxruntime


class TestRLPolicyControllerImport:
    def test_import(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        assert RLPolicyController is not None


class TestRLPolicyStaticHelpers:
    """Test static methods that don't require ONNX session."""

    def test_normalize_clip_range_tuple(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        assert RLPolicyController._normalize_clip_range((-5.0, 5.0)) == (-5.0, 5.0)

    def test_normalize_clip_range_scalar(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        assert RLPolicyController._normalize_clip_range(10.0) == (-10.0, 10.0)

    def test_normalize_clip_range_invalid(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        with pytest.raises(ValueError):
            RLPolicyController._normalize_clip_range("bad")

    def test_normalize_clip_range_inverted_raises(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        with pytest.raises(ValueError, match="lower bound"):
            RLPolicyController._normalize_clip_range((5.0, -5.0))

    def test_extract_feature_dim_int(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        assert RLPolicyController._extract_feature_dim([1, 166]) == 166

    def test_extract_feature_dim_string(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        assert RLPolicyController._extract_feature_dim(["batch", "166"]) == 166

    def test_extract_feature_dim_dynamic(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        assert RLPolicyController._extract_feature_dim(["N"]) is None

    def test_select_providers_cpu_only(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        providers = RLPolicyController._select_providers(
            lambda: ["CPUExecutionProvider"], "cpu"
        )
        assert providers == ["CPUExecutionProvider"]

    def test_select_providers_auto_with_cuda(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        providers = RLPolicyController._select_providers(
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"], "auto"
        )
        assert "CUDAExecutionProvider" in providers
        assert "CPUExecutionProvider" in providers

    def test_cfg_get_dict(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        assert RLPolicyController._cfg_get({"a": 1}, "a", 0) == 1

    def test_cfg_get_default(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        assert RLPolicyController._cfg_get({"a": 1}, "b", 99) == 99


class TestRLPolicyControllerInit:
    """Test init error paths."""

    def test_missing_policy_file_raises(self):
        from teleopit.controllers.rl_policy import RLPolicyController
        cfg = {"policy_path": "/nonexistent/model.onnx", "device": "cpu"}
        with pytest.raises(FileNotFoundError, match="ONNX policy file not found"):
            RLPolicyController(cfg)


@requires_onnxruntime
class TestRLPolicyControllerInference:
    """Test inference with a mocked ONNX session."""

    def test_compute_action_shape(self):
        from teleopit.controllers.rl_policy import RLPolicyController

        obs_dim = 166
        action_dim = 29

        # Build a controller with mocked internals
        ctrl = RLPolicyController.__new__(RLPolicyController)
        ctrl._expected_obs_dim = obs_dim
        ctrl.action_scale = np.ones(action_dim, dtype=np.float32)
        ctrl.default_dof_pos = np.zeros(action_dim, dtype=np.float32)
        ctrl.clip_range = (-10.0, 10.0)
        from collections import deque
        ctrl._history_length = 3
        ctrl._history_obs_dim = obs_dim
        ctrl._history_buf = deque(maxlen=3)
        ctrl._last_obs_input = None
        ctrl._last_obs_history_input = None

        # Mock session
        mock_session = MagicMock()
        mock_session.run.return_value = [np.zeros((1, action_dim), dtype=np.float32)]
        ctrl._session = mock_session
        ctrl._input_name = "obs"
        ctrl._output_name = "action"

        obs = np.zeros(obs_dim, dtype=np.float32)
        action = ctrl.compute_action(obs)
        assert action.shape == (action_dim,)
        assert action.dtype == np.float32

    def test_compute_action_wrong_dim_raises(self):
        from teleopit.controllers.rl_policy import RLPolicyController

        ctrl = RLPolicyController.__new__(RLPolicyController)
        ctrl._expected_obs_dim = 166
        ctrl.clip_range = (-10.0, 10.0)
        ctrl.action_scale = np.ones(29, dtype=np.float32)
        ctrl._session = MagicMock()
        ctrl._input_name = "obs"
        ctrl._output_name = "action"

        with pytest.raises(ValueError, match="dimension mismatch"):
            ctrl.compute_action(np.zeros(100, dtype=np.float32))

    def test_reset_is_noop(self):
        from teleopit.controllers.rl_policy import RLPolicyController

        ctrl = RLPolicyController.__new__(RLPolicyController)
        from collections import deque
        ctrl._history_buf = deque(maxlen=3)
        ctrl._last_obs_input = np.zeros(166, dtype=np.float32)
        ctrl._last_obs_history_input = np.zeros((3, 166), dtype=np.float32)
        assert ctrl.reset() is None
        assert len(ctrl._history_buf) == 0
        assert ctrl._last_obs_input is None
        assert ctrl._last_obs_history_input is None

    def test_multi_input_debug_inputs_capture_history(self):
        from teleopit.controllers.rl_policy import RLPolicyController

        ctrl = RLPolicyController.__new__(RLPolicyController)
        ctrl._expected_obs_dim = 166
        ctrl.clip_range = (-10.0, 10.0)
        ctrl.action_scale = np.ones(2, dtype=np.float32)
        ctrl.default_dof_pos = np.zeros(2, dtype=np.float32)
        ctrl._input_name = "obs"
        ctrl._output_name = "action"
        ctrl._multi_input = True
        ctrl._history_length = 3
        ctrl._history_obs_dim = 166
        from collections import deque
        ctrl._history_buf = deque(maxlen=3)
        ctrl._last_obs_input = None
        ctrl._last_obs_history_input = None
        mock_session = MagicMock()
        mock_session.run.return_value = [np.zeros((1, 2), dtype=np.float32)]
        ctrl._session = mock_session

        obs = np.zeros(166, dtype=np.float32)
        ctrl.compute_action(obs)
        debug = ctrl.get_debug_inputs()
        assert debug["obs"] is not None
        assert debug["obs_history"] is not None
        assert debug["obs_history"].shape == (3, 166)
