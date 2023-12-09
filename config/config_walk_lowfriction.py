
env_cls = "ICCGANHumanoidLowFriction"

env_params = dict(
    motion_file = "composite/assets/motions/clips_walk.yaml",
    contactable_links = ["right_foot", "left_foot"],
    ground_friction = 0.15
)
discriminators = {
    "walk/full": dict(
        key_links = None, parent_link = None,
    )
}

