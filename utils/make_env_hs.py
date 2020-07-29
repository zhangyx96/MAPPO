
def make_env(env_name, benchmark=False, discrete_action=False):
    from MAEE.mae_envs.envs.hide_and_seek import make_env
    from os.path import abspath, dirname, join


    core_dir = abspath(join(dirname(__file__), '..'))
    envs_dir = 'MAEE/mae_envs/envs'
    xmls_dir = 'xmls'
    env = make_env()
    return env



