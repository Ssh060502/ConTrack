"""One tracked object: fixed parameter conditions per parallel environment."""
from __future__ import annotations
import math
import torch


def parse_conditions(text):
    result = []
    for item in text.split(","):
        pieces = item.split(":")
        if len(pieces) != 2:
            raise ValueError("Use mass_multiplier:object_friction pairs, e.g. 1:2,3:2,1:0.7")
        mass, friction = map(float, pieces)
        if not all(math.isfinite(v) and v > 0 for v in (mass, friction)):
            raise ValueError("Mass multiplier and friction must be finite and > 0")
        result.append((mass, friction))
    if not result:
        raise ValueError("At least one condition required")
    return result


def configure_physics(base, conditions):
    if base.num_objects != 1:
        raise ValueError("Oracle adapter prototype supports exactly one tracked object")
    if base.num_envs < len(conditions):
        raise ValueError("Need at least one environment per parameter condition")
    view = base.scene["objects"].root_physx_view
    mass = view.get_masses().clone()
    inertia = view.get_inertias().clone()
    material = view.get_material_properties().clone()
    if mass.shape[0] != base.num_envs or inertia.shape[0] != base.num_envs or material.shape[0] != base.num_envs:
        raise ValueError("Unexpected PhysX collection layout; refusing to assign wrong environment parameters")
    labels = torch.arange(base.num_envs, device=mass.device) % len(conditions)
    pairs = torch.tensor(conditions, device=mass.device, dtype=mass.dtype)[labels]
    mass = mass * pairs[:, 0].reshape((-1,) + (1,)*(mass.ndim-1))
    inertia = inertia * pairs[:, 0].reshape((-1,) + (1,)*(inertia.ndim-1))
    material[..., 0] = pairs[:, 1].reshape((-1,) + (1,)*(material.ndim-2))
    material[..., 1] = material[..., 0]
    indices = torch.arange(base.num_envs, device=mass.device, dtype=torch.int32)
    view.set_masses(mass, indices)
    view.set_inertias(inertia, indices)
    view.set_material_properties(material, indices)
    actual_mass = view.get_masses().clone()
    actual_inertia = view.get_inertias().clone()
    actual_material = view.get_material_properties().clone()
    if not torch.allclose(actual_mass, mass, rtol=1e-4, atol=1e-8):
        raise RuntimeError("Mass readback mismatch")
    if not torch.allclose(actual_inertia, inertia, rtol=1e-4, atol=1e-8):
        raise RuntimeError("Inertia readback mismatch")
    if not torch.allclose(actual_material, material, rtol=1e-4, atol=1e-6):
        raise RuntimeError("Material readback mismatch")
    # All shapes of this one object must use the same coefficients.
    mu = actual_material[..., 0].reshape(base.num_envs, -1)
    if not torch.allclose(mu, mu[:, :1].expand_as(mu)):
        raise RuntimeError("Object shapes have inconsistent friction")
    kg = actual_mass.reshape(base.num_envs, -1)
    if kg.shape[1] != 1:
        raise ValueError("Expected one rigid-body mass per environment")
    context = torch.cat([kg, mu[:, :1]], dim=-1).to(base.device).log()
    report = {"conditions": conditions, "condition_id": labels.cpu().tolist(),
              "actual_mass_kg": actual_mass.cpu().tolist(),
              "actual_inertia": actual_inertia.cpu().tolist(),
              "material_static_dynamic_restitution": actual_material.cpu().tolist(),
              "oracle_features": "log(actual_mass_kg), log(object_material_static_friction)",
              "assignment": "fixed per environment for entire run; no per-episode parameter resampling"}
    return context, report


class OracleContextWrapper:
    def __init__(self, env, context):
        self.env, self.context = env, context

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def it_step_count(self):
        return self.env.it_step_count

    @it_step_count.setter
    def it_step_count(self, value):
        # The runner resets this counter after logging. Do not shadow the
        # reward wrapper's actual counter on this proxy.
        self.env.it_step_count = value

    def _augment(self, obs):
        result = obs.clone(recurse=False)
        result.set("oracle", self.context)
        return result

    def get_observations(self):
        return self._augment(self.env.get_observations())

    def step(self, actions):
        obs, reward, done, extra = self.env.step(actions)
        return self._augment(obs), reward, done, extra

    def reset(self):
        return self.env.reset()
