"""Isaac-only scene construction; imported after AppLauncher starts."""
from pathlib import Path

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.converters import UrdfConverterCfg


def build_scene(sim, bundle, manifest, qpos, poses, vertices, faces, args, output):
    """Independent 28-DOF articulations, placed once using recorded NPZ bases."""
    reference = args.mode == 'reference'
    light = sim_utils.DomeLightCfg(intensity=2500)
    light.func('/World/Light', light)
    xmin, xmax, ymin, ymax = manifest['table_bounds_xy']
    thickness = manifest['table_thickness']
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=args.friction, dynamic_friction=args.friction,
        restitution=0.0, friction_combine_mode='average', restitution_combine_mode='min')
    table = sim_utils.CuboidCfg(
        size=(xmax-xmin, ymax-ymin, thickness),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=not reference),
        physics_material=material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.35, .35, .38)))
    table.func('/World/Table', table, translation=(
        (xmin+xmax)/2, (ymin+ymax)/2, manifest['table_height']-thickness/2))

    robots = {}
    for index, side in enumerate(('left', 'right')):
        names = manifest['joint_names'][side]
        limits = manifest['joint_limits'][side]
        base = manifest['bases_xyz_wxyz'][side]
        importer = sim_utils.UrdfFileCfg(
            asset_path=str(bundle / side / 'robot.urdf'),
            usd_dir=str(output / 'usd_cache' / side), usd_file_name='robot.usd',
            force_usd_conversion=True, make_instanceable=False,
            fix_base=True, merge_fixed_joints=True,
            self_collision=args.self_collision and not reference,
            collider_type=args.collider,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type='force', target_type='position',
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=reference,
                                                        max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=args.self_collision and not reference,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4),
        )
        # Initial engineering PD values, NOT SDK MIT gains or a calibrated model.
        actuators = {}
        for label, subset, stiffness, damping in (
                ('arm', names[:6], args.arm_stiffness, args.arm_damping),
                ('hand', names[6:], args.hand_stiffness, args.hand_damping)):
            actuators[label] = ImplicitActuatorCfg(
                joint_names_expr=subset, stiffness=stiffness, damping=damping,
                effort_limit_sim={n: limits[n]['effort'] for n in subset},
                velocity_limit_sim={n: limits[n]['velocity'] for n in subset})
        robots[side] = Articulation(ArticulationCfg(
            prim_path=f'/World/{side.capitalize()}Robot', spawn=importer,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(base[:3]), rot=tuple(base[3:]),
                joint_pos=dict(zip(names, qpos[0, index*28:(index+1)*28]))),
            soft_joint_pos_limit_factor=1.0, actuators=actuators))

    # USD mesh is local object geometry, with world pose supplied independently.
    object_path = output / 'usd_cache' / 'hammer.usda'
    object_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(object_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/Hammer')
    stage.SetDefaultPrim(root.GetPrim())
    rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateKinematicEnabledAttr(reference)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(args.object_mass * args.mass_scale)
    mesh = UsdGeom.Mesh.Define(stage, '/Hammer/Mesh')
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDisplayColorAttr([Gf.Vec3f(.65, .32, .12)])
    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(not reference)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
        'convexHull' if args.collider == 'convex_hull' else 'convexDecomposition')
    stage.GetRootLayer().Save()
    hammer = RigidObject(RigidObjectCfg(
        prim_path='/World/Hammer', spawn=sim_utils.UsdFileCfg(usd_path=str(object_path)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(poses[0, :3]), rot=tuple(poses[0, 3:]))))

    # Override collision materials on BOTH hands/arms and the object; record the
    # common coefficient instead of implying object-only friction is effective mu.
    sim_stage = sim.stage
    material.func('/World/ContactMaterial', material)
    usd_material = UsdShade.Material.Get(sim_stage, '/World/ContactMaterial')
    for prim in sim_stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if reference:
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                usd_material, materialPurpose='physics')
    return robots, hammer
