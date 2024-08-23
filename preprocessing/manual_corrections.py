attack_corrections = {
    "A7": {
        ("ballthrowing / 100_cut", 1, 0, "right"): {"valid": True, "start": 7.5},
        ("ballthrowing / 100_cut", 2, 0, "right"): {"valid": True, "start": 7.5},
        ("ballthrowing / 100_cut", 3, 0, "right"): {"valid": True},
        ("ballthrowing / 102_cut", 1, 0, "right"): {"valid": False},
    },
}

main_corrections = {
    "adam32": {
        ("desk", 1, 0, "right"): {"end": 7.5},
        ("desk", 5, 0, "right"): {"end": 5.5},
        ("motion", 5, 0, "right"): {"end": 8},
    },
    "andrew10": {
        ("hand", 1, 0, "right"): {"end": 6.0},
        ("hand", 2, 0, "right"): {"end": 9.0},
    },
    "colonmichelle": {
        ("motion", 2, 1, "right"): {"valid": True},
    },
    "bryanburke": {
        ("password", 1, 0, "right"): {"start": 12.0},
    },
    "chad26": {
        ("motion", 1, 0, "right"): {"start": 1.0},
    },
    "daniel02": {
        ("match", 1, 1, "right"): {"valid": False},
    },
    "fwallace": {
        ("arrow", 1, 0, "right"): {"end": 9.0},
    },
    "gonzalezedwin": {
        ("level", 1, 0, "right"): {"end": 8.5},
    },
    "leah19": {
        ("oven", 5, 0, "right"): {"valid": False},
        ("password", 2, 0, "right"): {"valid": False},
        ("password", 3, 0, "right"): {"valid": False},
    },
    "lfisher": {("light", 1, 0, "right"): {"valid": False}},
    "rachel81": {
        ("motion", 3, 1, "right"): {"end": 10.5},
        ("queue", 1, 1, "right"): {"end": 7.0},
    },
    "rfleming": {
        ("basic", 4, 0, "right"): {"end": 6.0},
    },
    "susandeleon": {
        ("chance", 2, 0, "right"): {"end": 8.0},
    },
    "wendy61": {
        ("tire", 2, 0, "right"): {"end": 10.0},
    },
    "valdezkatie": {
        ("shout", 5, 0, "left"): {"valid": False},  # gibberish
        ("shout", 5, 0, "right"): {"valid": False},  # different style suddenly
        ("hungry", 4, 0, "right"): {"valid": False},  # random button presses at the end
        ("hungry", 5, 0, "right"): {"valid": False},  # random button presses at the end
    },
}
