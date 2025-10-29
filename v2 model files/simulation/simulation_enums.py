class SimulationConfigs:
    """
    Enum defining custom network configs for each setting
    """

    # Regular models
    CONFIG_0 = {
        "GLOBAL_NOISE_SCALE": 0.0,
        "NETWORK_MODIFIERS": {
            0: [0.2, 0.2, 1, 0.1, 0.1],
            1: [0.4, 0.2, 0.7, 0, 0.05],
            2: [0.1, 0.1, 0.6, 0.4, 0.05],
            3: [0.2, 0.4, 0.7, 0.3, 0.05],
            4: [0.1, 0.1, 0.9, 0.1, 0.1],
        },
    }

    CONFIG_1 = {
        "GLOBAL_NOISE_SCALE": 0.0,
        "NETWORK_MODIFIERS": {
            0: [0.05, 0.23, 1, 0.1, 0.1],
            1: [0.15, 0.2, 1, 0.2, 0.1],
            2: [0.6, 0.01, 1, 0.2, 0.1],
            3: [0.15, 0.23, 1, 0.2, 0.1],
            4: [0.05, 0.23, 1, 0.1, 0.1],
        },
    }

    # two maximum layers
    CONFIG_2 = {
        "GLOBAL_NOISE_SCALE": 0.0,
        "NETWORK_MODIFIERS": {
            0: [0.05, 0.25, 1, 0.1, 0.1],
            1: [0.15, 0.3, 1, 0.2, 0.1],
            2: [0.4, 0.1, 1, 0.2, 0.2],
            3: [0.4, 0.1, 1, 0.2, 0.2],
            4: [0.05, 0.25, 1, 0.1, 0.1],
        },
    }

    # linear increase/decr
    CONFIG_3 = {
        "GLOBAL_NOISE_SCALE": 0.0,
        "NETWORK_MODIFIERS": {
            0: [0.05, 0.5, 1, 0.2, 0.1],
            1: [0.1, 0.2, 1, 0.2, 0.1],
            2: [0.15, 0.15, 1, 0.2, 0.1],
            3: [0.2, 0.1, 1, 0.2, 0.1],
            4: [0.5, 0.05, 1, 0.2, 0.1],
        },
    }

    # linear increase/decr
    CONFIG_4 = {
        "GLOBAL_NOISE_SCALE": 0.0,
        "NETWORK_MODIFIERS": {
            0: [0.4, 0.1, 1, 0.2, 0.2],
            1: [0.05, 0.25, 1, 0.1, 0.1],
            2: [0.15, 0.3, 1, 0.2, 0.1],
            3: [0.05, 0.25, 1, 0.1, 0.1],
            4: [0.4, 0.1, 1, 0.2, 0.2],
        },
    }

    # format
    # { network_idx:
    #   [
    #       ne_contrib_weight_idx,
    #       qps_contrib_weight_idx,
    #       flops_scale_idx,
    #       flax_scale_idx,
    #       depth_scale_idx
    #   ]
    # }
