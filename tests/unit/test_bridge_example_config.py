# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise recipe overrides without loading a training runtime."""

import importlib.util
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "bridge_example_config",
    Path(__file__).resolve().parents[2] / "examples/megatron_bridge/config_overrides.py",
)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)


class Backend(Enum):
    auto = 1


def test_nested_overrides_preserve_enum_and_native_types(tmp_path):
    path = tmp_path / "overrides.yaml"
    path.write_text("model:\n  backend: auto\noptimizer:\n  offload: true\n")
    config = SimpleNamespace(
        model=SimpleNamespace(backend=Backend.auto), optimizer=SimpleNamespace(offload=False)
    )
    _config.apply_overrides(config, _config.read_overrides(path))
    assert config.model.backend is Backend.auto
    assert config.optimizer.offload is True


@pytest.mark.parametrize("invalid", [{"missing": True}, {"count": False}])
def test_invalid_override_does_not_partially_mutate_config(invalid):
    config = SimpleNamespace(enabled=False, count=2)
    with pytest.raises(ValueError):
        _config.apply_overrides(config, {"enabled": True, **invalid})
    assert config.enabled is False and config.count == 2
