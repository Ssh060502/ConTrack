"""Robot selection shared by adapter workflows, with no simulator imports."""
import re


def robot_profile(robot):
    if robot == 'rby1a':
        return dict(task='Isaac-Rby1a-Sharpa-Mimic-Manager-v0',
                    data_env='CONTRACK_RBY1A_SHARPA_DATA',
                    mdp='ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp')
    if robot == 'crx':
        return dict(task='Isaac-Crx-Sharpa-Mimic-Manager-v0',
                    data_env='CONTRACK_CRX_SHARPA_DATA',
                    mdp='ConTrack.tasks.manager_based.mimic_crx_sharpa.mdp')
    raise ValueError(f'Unknown robot: {robot}')


def action_indices(names):
    arms = [i for i,n in enumerate(names) if re.fullmatch(r'(left|right)_(arm_[0-6]|J[1-6])', n)]
    fingers = [i for i in range(len(names)) if i not in arms]
    if not arms or not fingers:
        raise ValueError('Cannot determine named arm/finger action mapping')
    return arms, fingers


def collection_robot(metadata):
    task = metadata.get('task', '')
    if 'Crx-Sharpa' in task:
        return 'crx'
    if 'Rby1a-Sharpa' in task:
        return 'rby1a'
    # Legacy collections lack task metadata. New CRX filenames are explicit.
    return 'crx' if 'crx_sharpa' in metadata.get('reference_path', '').lower() else 'rby1a'


def history_state_indices(names, mode='all'):
    if mode == 'all':
        return list(range(len(names)))
    if mode != 'joints':
        raise ValueError(f'Unknown history state feature mode: {mode}')
    ids=[i for i,name in enumerate(names) if '/qpos/' in name or '/qvel/' in name]
    if not ids:
        raise ValueError('No joint position/velocity history features')
    return ids
