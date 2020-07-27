
def make_env(env_name, benchmark=False, discrete_action=False):
    from MAEE.mae_envs.viewer.env_viewer import EnvViewer
    from MAEE.mae_envs.wrappers.multi_agent import JoinMultiAgentActions
    from mujoco_worldgen.util.envs import examine_env, load_env
    from mujoco_worldgen.util.types import extract_matching_arguments
    from mujoco_worldgen.util.parse_arguments import parse_arguments
    from gym.spaces import Tuple
    from MAEE.mae_envs.envs.hide_and_seek import make_env
    import logging
    import click
    import numpy as np
    from os.path import abspath, dirname, join

    core_dir = abspath(join(dirname(__file__), '..'))
    envs_dir = 'MAEE/mae_envs/envs'
    xmls_dir = 'xmls'
    env = make_env()
    return env



