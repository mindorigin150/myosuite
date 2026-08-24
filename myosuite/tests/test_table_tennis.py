from importlib.resources import files

import gymnasium as gym
import mujoco
import myosuite  # noqa: F401
import numpy as np
from myosuite.envs.myo.myochallenge.tabletennis_v0 import (
    ContactTrajIssue,
    PingpongContactLabels,
    TableTennisEnvV0,
    evaluate_pingpong_trajectory,
    preprocess_table_tennis_spec,
)


def test_p2_preprocessor_preserves_official_paddle_geometry():
    model_path = (
        files("myosuite") / "envs" / "myo" / "assets" / "arm" / "myoarm_tabletennis.xml"
    )
    spec = preprocess_table_tennis_spec(mujoco.MjSpec.from_file(str(model_path)))
    model = spec.compile()

    assert (model.nq, model.nv, model.nu, model.na) == (72, 70, 275, 273)
    pad_id = model.geom("pad").id
    assert model.geom_type[pad_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
    np.testing.assert_array_equal(model.geom_size[pad_id], [0.093, 0.020, 0.0])
    assert model.geom_contype[pad_id] == 1
    assert model.geom_conaffinity[pad_id] == 1
    default_option = mujoco.MjOption()
    mujoco.mj_defaultOption(default_option)
    assert model.opt.ccd_tolerance == default_option.ccd_tolerance
    assert model.opt.tolerance == default_option.tolerance


def test_p2_reset_seed_covers_domain_randomization():
    env = gym.make("myoChallengeTableTennisP2-v0")
    first_obs, _ = env.reset(seed=7)
    first_mass = env.unwrapped.mj_model.body_mass[env.unwrapped.id_info.paddle_bid]
    first_subtree_mass = env.unwrapped.mj_model.body_subtreemass[
        env.unwrapped.id_info.paddle_bid
    ]
    first_friction = env.unwrapped.mj_model.geom_friction[
        env.unwrapped.id_info.ball_gid
    ].copy()

    second_obs, _ = env.reset(seed=7)
    second_mass = env.unwrapped.mj_model.body_mass[env.unwrapped.id_info.paddle_bid]
    second_friction = env.unwrapped.mj_model.geom_friction[
        env.unwrapped.id_info.ball_gid
    ]

    np.testing.assert_array_equal(second_obs, first_obs)
    assert second_mass == first_mass
    assert first_subtree_mass == first_mass
    np.testing.assert_array_equal(second_friction, first_friction)


def test_contact_trajectory_issues_terminate():
    env = object.__new__(TableTennisEnvV0)
    trajectories = {
        ContactTrajIssue.NO_PADDLE: [{PingpongContactLabels.OPPONENT}],
        ContactTrajIssue.OWN_HALF: [
            {PingpongContactLabels.OWN},
            set(),
            {PingpongContactLabels.OWN},
        ],
        ContactTrajIssue.DOUBLE_TOUCH: [
            {PingpongContactLabels.PADDLE},
            set(),
            {PingpongContactLabels.PADDLE},
        ],
    }

    for expected_issue, trajectory in trajectories.items():
        env.obs_dict = {"time": 0.0}
        env.contact_trajectory = trajectory
        assert evaluate_pingpong_trajectory(trajectory) is expected_issue
        assert env._get_done(z=1.0, solved=False) == 1

    env.obs_dict = {"time": 0.0}
    env.contact_trajectory = [{PingpongContactLabels.PADDLE}]
    assert evaluate_pingpong_trajectory(env.contact_trajectory) is ContactTrajIssue.MISS
    assert env._get_done(z=1.0, solved=True) == 1
