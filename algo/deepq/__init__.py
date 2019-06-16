from algo.deepq import models  # noqa
from algo.deepq.build_graph import build_act, build_train  # noqa
from algo.deepq.deepq import learn, load_act  # noqa
from algo.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from algo.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
