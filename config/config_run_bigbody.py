
env_cls = "ICCGANHumanoidTarget"

env_params = dict(
    character_model = "assets/humanoid_bigbody.xml",

    motion_file = "composite/assets/motions/clips_run.yaml",
    contactable_links = ["right_foot", "left_foot"],

    goal_radius = 0.5,
    sp_lower_bound = 2,
    sp_upper_bound = 4,
    goal_timer_range = (60, 90),
    goal_sp_mean = 1.5,
    goal_sp_std = 0.5,
    goal_sp_min = 1,
    goal_sp_max = 3
)

discriminators = {
    "run/full": dict(
        key_links = None, parent_link = None,
    )
}
