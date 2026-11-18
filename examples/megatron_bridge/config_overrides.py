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

"""Validated overrides and optional callbacks for the common Bridge examples."""

import importlib
from enum import Enum
from pathlib import Path

import yaml


def read_overrides(path):
    """Read a trusted local YAML mapping; an omitted file leaves defaults unchanged."""
    if path is None:
        return {}
    result = yaml.safe_load(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError("Config overrides must be a YAML mapping.")
    return result


def apply_overrides(target, overrides):
    """Validate all field paths before changing an existing configuration object."""
    updates = []

    def collect(obj, values, prefix=""):
        if not isinstance(values, dict):
            raise ValueError(f"Expected a mapping at {prefix or 'config'}.")
        for key, value in values.items():
            name = f"{prefix}.{key}" if prefix else key
            if key.startswith("_") or not hasattr(obj, key):
                raise ValueError(f"Unknown config field: {name}")
            previous = getattr(obj, key)
            if isinstance(value, dict) and previous is not None and not isinstance(previous, dict):
                collect(previous, value, name)
                continue
            if isinstance(previous, Enum):
                value = (
                    type(previous)[value]
                    if isinstance(value, str) and value in type(previous).__members__
                    else type(previous)(value)
                )
            elif previous is not None and not isinstance(value, type(previous)):
                if not (isinstance(previous, float) and type(value) is int):
                    raise ValueError(f"Invalid type for config field: {name}")
            if type(previous) is int and type(value) is bool:
                raise ValueError(f"Invalid type for config field: {name}")
            updates.append((obj, key, value))

    collect(target, overrides)
    for obj, key, value in updates:
        setattr(obj, key, value)


def build_callbacks(specifications):
    """Instantiate explicit callback factories from the caller's trusted local recipe."""
    callbacks = []
    for specification in specifications:
        values = dict(specification)
        target = values.pop("_target_")
        module, separator, name = target.rpartition(".")
        if not separator:
            raise ValueError("Callback _target_ must be a fully qualified name.")
        factory = getattr(importlib.import_module(module), name)
        callbacks.append(factory(**values))
    return callbacks
