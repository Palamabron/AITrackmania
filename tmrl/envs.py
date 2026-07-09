# standard library imports

# third-party imports
import gymnasium

# local imports
from tmrl.wrappers import AffineObservationWrapper, Float64ToFloat32

__docformat__ = "google"


class GenericGymEnv(gymnasium.Wrapper):
    """Lightweight Gymnasium wrapper for use with the TMRL framework.

    Optionally applies an affine observation rescaling and/or a float32 cast.
    Use this when integrating arbitrary Gymnasium environments with TMRL.
    """

    def __init__(
        self, id: str = "Pendulum-v0", gym_kwargs=None, obs_scale: float = 0.0, to_float32=False
    ):
        """
        Use this wrapper when using the framework with arbitrary environments.

        Args:
            id (str): gymnasium id
            gym_kwargs (dict): keyword arguments for the gymnasium environment
            obs_scale (float): rescale actions by this scalar if set
            to_float32 (bool): if True, convert observations to numpy.float32
        """
        if gym_kwargs is None:
            gym_kwargs = {}
        env = gymnasium.make(id, **gym_kwargs, disable_env_checker=True)
        if obs_scale:
            env = AffineObservationWrapper(env, 0, obs_scale)
        if to_float32:
            env = Float64ToFloat32(env)
        super().__init__(env)
