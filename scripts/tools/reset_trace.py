"""Read-only reset/first-control diagnostics. No extra physics or scene updates."""
from __future__ import annotations

import json
import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def write_snapshot(group, values):
    for key, value in values.items():
        value = np.asarray(value)
        group.create_dataset(key, data=value, **({"compression": "gzip"} if value.ndim else {}))


class ResetTrace:
    """Captures full articulation buffers plus fresh PhysX DOFs at each substep.

    Buffered articulation data can be stale immediately after sim.step(), before
    scene.update(). Therefore post-substep comparisons must use physx/* fields.
    Optional version-specific getters are reported as unavailable, not invented.
    """

    def __init__(self, env):
        self.env = env
        self.original_step = env.sim.step
        self.records = {}
        self.reset_index = -1
        self.current = None
        self.unavailable = {}
        self.robots = {name: env.scene[name] for name, _, _ in env.ref_hands}

    def _optional(self, result, key, owner, attribute, call=False, **kwargs):
        try:
            value = getattr(owner, attribute)
            if call:
                value = value(**kwargs)
            if value is None:
                raise AttributeError("returned None")
            result[key] = array(value)
        except (AttributeError, RuntimeError, NotImplementedError) as exc:
            self.unavailable[key] = f"{type(exc).__name__}: {exc}"

    def snapshot(self):
        result = {}
        for name, robot in self.robots.items():
            prefix = f"robots/{name}"
            data = robot.data
            # All joints, including those absent from the reference/action vector.
            for attribute in ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w",
                              "root_lin_vel_w", "root_ang_vel_w", "joint_pos_target",
                              "joint_vel_target", "joint_effort_target"):
                result[f"{prefix}/buffered/{attribute}"] = array(getattr(data, attribute))
            for attribute in ("computed_torque", "applied_torque", "joint_acc", "body_state_w"):
                self._optional(result, f"{prefix}/buffered/{attribute}", data, attribute)
            view = robot.root_physx_view
            # Required fresh values: fail visibly if the server API differs.
            for getter in ("get_dof_positions", "get_dof_velocities",
                           "get_link_transforms", "get_link_velocities"):
                result[f"{prefix}/physx/{getter}"] = array(getattr(view, getter)())
            for getter in ("get_root_transforms", "get_root_velocities", "get_dof_position_targets",
                           "get_dof_velocity_targets", "get_dof_actuation_forces"):
                self._optional(result, f"{prefix}/physx/{getter}", view, getter, call=True)
        view = self.env.scene["objects"].root_physx_view
        for getter in ("get_transforms", "get_velocities"):
            result[f"objects/physx/{getter}"] = array(getattr(view, getter)())
        result["reference_frame"] = array(self.env.frame_idx)
        result["action_manager_input"] = array(self.env.action_manager.action)
        # Read the existing cache directly: sensor.data would trigger a lazy update.
        for side, keys in getattr(self.env, "hand_contact_sensor_keys", {}).items():
            for index, name in enumerate(keys):
                sensor = self.env.scene[name]
                prefix = f"contact/{side}/{index}"
                cache = getattr(sensor, "_data", None)
                for attribute in ("net_forces_w", "force_matrix_w"):
                    self._optional(result, f"{prefix}/cached/{attribute}", cache, attribute)
                try:
                    view = sensor.contact_physx_view
                except (AttributeError, RuntimeError) as exc:
                    self.unavailable[f"{prefix}/physx"] = str(exc)
                    continue
                for getter in ("get_net_contact_forces", "get_contact_force_matrix"):
                    self._optional(result, f"{prefix}/physx/{getter}", view, getter,
                                   call=True, dt=float(self.env.cfg.sim.dt))
        return result

    def reset_before(self):
        self.reset_index += 1
        record = {"reset_before": self.snapshot(), "substeps": []}
        self.records[self.reset_index] = record

    def reset_after(self):
        self.records[self.reset_index]["reset_after"] = self.snapshot()

    def before_policy(self, episode):
        record = self.records[episode]
        if "before_policy" not in record:
            record["before_policy"] = self.snapshot()

    def begin_control(self, episode):
        record = self.records[episode]
        self.current = record if "first_control_end" not in record else None
        if self.current is not None:
            self.current["after_policy_before_step"] = self.snapshot()

    def end_control(self):
        if self.current is not None:
            # On a terminal first step this is a post-reset state, explicitly named.
            self.current["first_control_end"] = self.snapshot()
        self.current = None

    def install(self):
        def step(*args, **kwargs):
            record = self.current
            if record is None:
                return self.original_step(*args, **kwargs)
            before = self.snapshot()
            value = self.original_step(*args, **kwargs)
            after = self.snapshot()
            record["substeps"].append({"before": before, "after": after})
            return value
        self.env.sim.step = step

    def restore(self):
        self.env.sim.step = self.original_step

    def write(self, group, episode):
        record = self.records.pop(episode)
        group.attrs["physics_dt"] = float(self.env.cfg.sim.dt)
        group.attrs["decimation"] = int(self.env.cfg.decimation)
        group.attrs["unavailable_fields_json"] = json.dumps(self.unavailable)
        group.attrs["joint_names_json"] = json.dumps({k: list(v.joint_names) for k, v in self.robots.items()})
        group.attrs["link_names_json"] = json.dumps({k: list(v.body_names) for k, v in self.robots.items()})
        group.attrs["trace_version"] = 3
        group.attrs["contact_mapping_json"] = json.dumps(getattr(self.env, "hand_contact_sensor_keys", {}))
        group.attrs["contact_force_semantics"] = "PhysX force = impulse / physics_dt; cached sensor buffers may be stale. Filtered matrix covers configured filters only; total force does not identify the other body."
        group.attrs["link_transform_semantics"] = "PhysX native (x,y,z,qx,qy,qz,qw); frame follows simulation subspace; no kinematic refresh is forced"
        group.attrs["link_velocity_semantics"] = "PhysX world linear velocity at link COM followed by angular velocity"
        group.attrs["buffer_semantics"] = "buffered fields may be stale between sim.step and scene.update; use physx for fresh values"
        group.attrs["raw_transform_semantics"] = "native PhysX tensor API layout and world coordinates, no quaternion conversion"
        for phase in ("reset_before", "reset_after", "before_policy", "after_policy_before_step", "first_control_end"):
            if phase in record:
                write_snapshot(group.create_group(phase), record[phase])
        if len(record["substeps"]) != int(self.env.cfg.decimation):
            raise RuntimeError(f"Expected {self.env.cfg.decimation} first-control physics steps, captured {len(record['substeps'])}")
        for index, step in enumerate(record["substeps"]):
            for phase in ("before", "after"):
                write_snapshot(group.create_group(f"substeps/{index:03d}/{phase}"), step[phase])
