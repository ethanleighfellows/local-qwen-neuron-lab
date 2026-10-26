"""Real down-projection input interventions; never modifies model parameters."""
import re
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Code:
    layer: int
    neuron: int
    dose: float
    enabled: bool = True


def parse_codes(text, layers, width):
    codes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = re.fullmatch(r"L(\d+)\s+(\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))", line.strip())
        if not m:
            raise ValueError(f"Invalid steering code: {line}")
        layer, neuron, dose = int(m[1]), int(m[2]), float(m[3])
        if not 0 <= layer < layers or not 0 <= neuron < width:
            raise ValueError(f"Out of range: {line}; architecture is {layers} x {width}")
        if any(c.layer == layer and c.neuron == neuron for c in codes):
            raise ValueError("Duplicate neuron; edit its dose instead")
        codes.append(Code(layer, neuron, dose))
    return codes


class SteeringEngine:
    def __init__(self, layers, residual_norms):
        self.layers = layers
        self.residual_norms = residual_norms
        self.column_norms = {}
        self.handles = []
        self.shifts = {}
        self.observations = []

    def column_norm(self, layer, neuron):
        key = (layer, neuron)
        if key not in self.column_norms:
            weight = self.layers[layer].mlp.down_proj.weight
            if getattr(weight, "quant_state", None) is not None:
                from bitsandbytes.functional import dequantize_4bit
                matrix = dequantize_4bit(weight.data, weight.quant_state)
            else:
                matrix = weight.detach()
            norm = matrix[:, neuron].float().norm().item()
            if norm <= 0:
                raise ValueError("Zero down-projection column norm")
            self.column_norms[key] = norm
        return self.column_norms[key]

    def install(self, codes):
        self.remove()
        self.observations = []
        grouped = {}
        for c in codes:
            if c.enabled and c.dose != 0:
                shift = c.dose * self.residual_norms[c.layer] / self.column_norm(c.layer, c.neuron)
                self.shifts[f"L{c.layer} {c.neuron:05d}"] = shift
                grouped.setdefault(c.layer, []).append((c.neuron, shift))
        for layer, shifts in grouped.items():
            def hook(module, args, layer=layer, shifts=tuple(shifts)):
                original = args[0]
                changed = original.clone()
                for neuron, shift in shifts:
                    changed[..., neuron] += shift
                    self.observations.append({"layer": layer, "neuron": neuron,
                        "tokens": original.shape[-2], "requested_shift": shift,
                        "measured_shift": (changed[..., neuron] - original[..., neuron]).float().mean().item()})
                return (changed,) + args[1:]
            self.handles.append(self.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.shifts.clear()
