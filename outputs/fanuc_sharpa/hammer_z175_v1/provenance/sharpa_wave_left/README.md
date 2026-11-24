# Sharpa Wave left hand

Self-contained copy of the left hand with flange: 22 revolute joints,
35 links, no mimic joints, and 25 referenced STL meshes.

- URDF entry point: [urdf/left_sharpa_wave_with_flange.urdf](urdf/left_sharpa_wave_with_flange.urdf).
- Meshes: `meshes/`, addressed relative to the URDF as `../meshes/`.
- Source: `sharpa-robotics/sharpa-urdf-usd-xml`, revision `0d19cac602f46456b819e4b6a2c09a74982c9a3e`.
- Imported from the pinned model bundle in `dual_sharpa_wave_ros2`.
  That repository is only the import source; it is not a runtime dependency.
- The only URDF modification is replacing the mesh URI prefix with
  `../meshes/`. Joint names, axes, limits, transforms, inertias, and geometry
  are unchanged. Mesh files are copied byte-for-byte; there are no symlinks.
- `manifest.yaml` records upstream and imported URDF provenance.
  `SHA256SUMS.txt` records checksums of this local bundle.
- Original `LICENSE.txt` and `NOTICE.txt` are preserved unchanged.
  `Apache-2.0.txt` contains the full license text copied from the source package.

This is a standalone hand asset, not a combined CRX model. The CRX-to-hand
mounting transform and retargeting profile remain to be configured.
