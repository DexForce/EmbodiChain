"""Task predicates extracted from WorldParticle TrajDataset; no dataset imports."""

OBSTACLE_NPZ_DATA = {
    "sand",
    "elastic",
    "cloth_on_ball",
    "cloth_on_needle",
    "elastic_drag",
    "elastic_drop",
    "non",
    "non_obj",
    "non_bunny",
    "fancy_fluid",
    "cloth_no_self_attention",
    "cloth_no_ffn",
    "cloth_no_all",
    "fancy_flulid_div",
    "fancy_fluid_div",
    "real_world",
    "real_world_override",
    "duck_drop",
    "hair_ball",
}

TOPOLOGY_ENABLED_DATA = {
    "elastic",
    "cloth_on_ball",
    "cloth_no_self_attention",
    "cloth_no_ffn",
    "cloth_no_all",
    "cloth_on_needle",
    "elastic_drag",
    "elastic_drop",
    "twist",
    "twist_force",
    "twist_force_override",
    "duck_drop",
    "hair_ball",
}

TOPOLOGY_DEFAULT_SEARCH_RADIUS_BY_DATA = {
    "flag": 0.05,
    "cloth": 0.05,  # backward-compatible alias
    "elastic": 0.03,
    "cloth_on_needle": 0.005,
    "cloth_on_ball": 0.01,
    "cloth_no_self_attention": 0.01,
    "cloth_no_ffn": 0.01,
    "cloth_no_all": 0.01,
    "elastic_drag": 0.01,
    "elastic_drop": 0.01,
    "twist": 0.007,
    "twist_force": 0.007,
    "twist_force_override": 0.007,
    "duck_drop": 0.004,
    "hair_ball": 0.008,
}

OBSTACLE_DATA_TYPES = {"fluid"} | OBSTACLE_NPZ_DATA


def uses_topology_neighbors(data):
    return data in TOPOLOGY_ENABLED_DATA


def get_default_topology_search_radius(data):
    return TOPOLOGY_DEFAULT_SEARCH_RADIUS_BY_DATA.get(data, None)


def uses_obstacle_features(data):
    return data in OBSTACLE_DATA_TYPES
