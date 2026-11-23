"""Isaac-only scene construction; imported after AppLauncher starts."""
from pathlib import Path

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from fanuc_scene_data import initial_joint_positions, stage_import_urdf


def verify_imported_usd(path, side):
    """Fail at conversion instead of spawning an unresolved USD reference."""
    stage = Usd.Stage.Open(str(path))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f'{side}: URDF conversion produced no valid defaultPrim: {path}')
    if not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()):
        raise RuntimeError(f'{side}: converted USD has no articulation root: {path}')


def scene_visual_report(stage):
    """Inspect actual composed meshes, rather than assuming joints imply visuals."""
    report = {}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    for side in ('left', 'right'):
        root = stage.GetPrimAtPath(f'/World/{side.capitalize()}Robot')
        entries = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            imageable = UsdGeom.Imageable(prim)
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            entries.append({
                'path': str(prim.GetPath()), 'points': 0 if points is None else len(points),
                'visibility': str(imageable.ComputeVisibility()),
                'purpose': str(imageable.ComputePurpose()),
                'bounds_min': None if bounds.IsEmpty() else list(bounds.GetMin()),
                'bounds_max': None if bounds.IsEmpty() else list(bounds.GetMax()),
            })
        arm = [e for e in entries if any(
            f'/{name}/' in e['path'] for name in ['base_link'] + [f'J{i}_link' for i in range(1, 7)])]
        visible = [e for e in arm if e['points'] and e['visibility'] != 'invisible'
                   and e['purpose'] in ('default', 'render')]
        report[side] = {'mesh_count': len(entries), 'arm_mesh_count': len(arm),
                        'visible_arm_mesh_count': len(visible), 'meshes': entries}
        print(f'[fanuc] {side}: meshes={len(entries)}, arm meshes={len(arm)}, '
              f'render-visible arm meshes={len(visible)}', flush=True)
    return report


def build_scene(sim, bundle, manifest, qpos, poses, vertices, faces, args, output):
    """Independent 28-DOF articulations, placed once using recorded NPZ bases."""
    reference = args.mode == 'reference'
    # The CRX links are white. A camera-visible 2500-unit dome gave a white,
    # overexposed frame in which they could not be seen; keep the dome out of
    # primary rays (dark background) and add a directional light for shading.
    light = sim_utils.DomeLightCfg(intensity=800., visible_in_primary_ray=False)
    light.func('/World/Light', light)
    sun = sim_utils.DistantLightCfg(intensity=1200., angle=2.)
    sun.func('/World/KeyLight', sun, orientation=(.940, .342, 0., 0.))
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
        staged = stage_import_urdf(bundle / side / 'robot.urdf',
                                   output / 'import_source' / side / 'robot.urdf',
                                   collision_visuals=args.collision_visuals)
        importer = sim_utils.UrdfFileCfg(
            asset_path=str(staged), root_link_name='base_link',
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
        print(f'[fanuc] Converting {side}: {staged}', flush=True)
        converted = UrdfConverter(importer)
        verify_imported_usd(converted.usd_path, side)
        print(f'[fanuc] {side}: converted USD validated', flush=True)
        spawn = sim_utils.UsdFileCfg(
            usd_path=converted.usd_path, rigid_props=importer.rigid_props,
            articulation_props=importer.articulation_props)
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
            prim_path=f'/World/{side.capitalize()}Robot', spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(base[:3]), rot=tuple(base[3:]),
                joint_pos=initial_joint_positions(names, qpos[0, index*28:(index+1)*28], limits)),
            soft_joint_pos_limit_factor=1.0, actuators=actuators))
        print(f'[fanuc] {side}: articulation spawned', flush=True)

    # USD mesh is local object geometry, with world pose supplied independently.
    object_path = output / 'usd_cache' / 'hammer.usda'
    object_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(object_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/Hammer')
    stage.SetDefaultPrim(root.GetPrim())
    rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    # Never kinematic: reference playback teleports a gravity-free dynamic body
    # and steps, the same state-write path used for the articulations.
    rigid.CreateKinematicEnabledAttr(False)
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
    print('[fanuc] Creating hammer rigid object', flush=True)
    hammer = RigidObject(RigidObjectCfg(
        prim_path='/World/Hammer', spawn=sim_utils.UsdFileCfg(
            usd_path=str(object_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=reference)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(poses[0, :3]), rot=tuple(poses[0, 3:]))))

    # Override collision materials on BOTH hands/arms and the object; record the
    # common coefficient instead of implying object-only friction is effective mu.
    sim_stage = sim.stage
    print('[fanuc] Binding scene contact materials', flush=True)
    material.func('/World/ContactMaterial', material)
    usd_material = UsdShade.Material.Get(sim_stage, '/World/ContactMaterial')
    for prim in sim_stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if reference:
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                usd_material, materialPurpose='physics')
    print('[fanuc] Scene construction complete', flush=True)
    return robots, hammer
