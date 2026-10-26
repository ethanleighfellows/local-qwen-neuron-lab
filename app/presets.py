"""Bundled profiles installed on first run without replacing user edits."""
from copy import deepcopy


PRESETS = {
    "L14 N8949 +0.4": {
        "enabled": True,
        "steering": [
            {"layer": 14, "neuron": 8949, "dose": 0.4, "enabled": True},
        ],
    },
    "Full dose": {
        "enabled": True,
        "steering": [
            {"layer": 14, "neuron": 8949, "dose": 1.0, "enabled": True},
            {"layer": 13, "neuron": 7956, "dose": -1.0, "enabled": True},
        ],
    },
    "Half dose": {
        "enabled": True,
        "steering": [
            {"layer": 14, "neuron": 8949, "dose": 0.5, "enabled": True},
            {"layer": 13, "neuron": 7956, "dose": -0.5, "enabled": True},
        ],
    },
}


def initial_state():
    return {
        "active_profile": "Baseline", "enabled": True, "version": 0,
        "profiles": {"Baseline": {"enabled": False, "steering": []}, **deepcopy(PRESETS)},
    }
