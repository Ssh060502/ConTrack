"""Read authored robot collision settings without starting Isaac Sim or hardware.

Run in an environment providing pxr. This inspects the asset, not the live
PhysX collision cooking or runtime overrides applied by an environment.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from pxr import Usd, UsdGeom, UsdPhysics

    path = Path(args.usd).expanduser().resolve()
    if not path.is_file():
        parser.error(f"USD not found: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {path}")
    collisions = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        entry = {"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                 "applied_schemas": list(prim.GetAppliedSchemas()), "attributes": {}}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if any(token in name.lower() for token in
                   ("collision", "approximation", "contactoffset", "restoffset")):
                value = attr.Get()
                entry["attributes"][name] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            entry["vertex_count"] = len(points) if points is not None else 0
            entry["face_count"] = len(counts) if counts is not None else 0
            if points is not None and len(points):
                entry["mesh_local_min"] = [min(float(p[i]) for p in points) for i in range(3)]
                entry["mesh_local_max"] = [max(float(p[i]) for p in points) for i in range(3)]
        collisions.append(entry)
    report = {"asset": str(path), "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
              "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
              "layers": [layer.identifier for layer in stage.GetUsedLayers()],
              "collision_count": len(collisions), "collisions": collisions,
              "scope": "Composed asset collision attributes only; not live scene or cooked PhysX geometry"}
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Collision prims: {len(collisions)}; report: {output}")
    for entry in collisions:
        if "pinky" in entry["path"].lower() and "right" in entry["path"].lower():
            print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
