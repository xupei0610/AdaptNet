
env_cls = "ICCGANHumanoidTerrain"

env_params = dict(
    motion_file = "composite/assets/motions/clips_walk.yaml",
    contactable_links = ["right_foot", "left_foot"]
)
discriminators = {
    "walk/full": dict(
        key_links = None, parent_link = None,
    )
}

