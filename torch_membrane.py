"""
Condensate: PyTorch Membrane

Hooks into nn.Module forward passes to track which layers,
attention heads, and weight regions activate per input.
This is the real membrane — not wrapping dicts, but wrapping
model inference.

Works with any HuggingFace transformers model.

Usage:
    from torch_membrane import TorchMembrane

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    membrane = TorchMembrane(model)

    # Run inference — membrane records everything
    output = model.generate(input_ids)

    # See what activated
    membrane.print_activation_map()

    # Get the access log for graph building
    log = membrane.to_access_log()
"""

import time
import numpy as np
from collections import defaultdict


class LayerActivation:
    """Records activation statistics for a single layer."""

    __slots__ = ['name', 'forward_count', 'total_activation',
                 'max_activation', 'output_norms', 'timestamps_ns',
                 'param_bytes', 'is_attention', 'head_activations']

    def __init__(self, name, param_bytes=0, is_attention=False, num_heads=0):
        self.name = name
        self.forward_count = 0
        self.total_activation = 0.0
        self.max_activation = 0.0
        self.output_norms = []
        self.timestamps_ns = []
        self.param_bytes = param_bytes
        self.is_attention = is_attention
        # Per-head tracking for attention layers
        self.head_activations = [0.0] * num_heads if num_heads > 0 else []


class TorchMembrane:
    """Hooks into a PyTorch model to track layer activations.

    Installs forward hooks on every module. Records:
    - Which layers fire (have non-trivial output)
    - Output norm per layer (activation intensity)
    - Timing between layer activations (for causal chains)
    - Per-head activation for attention layers
    """

    def __init__(self, model, activation_threshold=0.01):
        """
        Args:
            model: nn.Module (typically a HuggingFace model)
            activation_threshold: minimum output norm to count as "active"
        """
        self.model = model
        self.activation_threshold = activation_threshold
        self.layers = {}           # name → LayerActivation
        self._hooks = []
        self._start_time = time.monotonic_ns()
        self._access_log = []      # [(timestamp_ns, event_type, path, size_bytes)]

        self._install_hooks()

    def _install_hooks(self):
        """Install forward hooks on all modules."""
        import torch

        for name, module in self.model.named_modules():
            if name == '':
                continue  # skip root

            # Count parameters
            param_bytes = sum(p.numel() * p.element_size()
                             for p in module.parameters(recurse=False))

            # Detect attention layers
            is_attention = any(kw in name.lower()
                              for kw in ['attn', 'attention', 'self_attn'])
            num_heads = getattr(module, 'num_heads',
                       getattr(module, 'num_attention_heads', 0))

            layer_info = LayerActivation(
                name=name,
                param_bytes=param_bytes,
                is_attention=is_attention,
                num_heads=num_heads,
            )
            self.layers[name] = layer_info

            # Install hook
            hook = module.register_forward_hook(
                self._make_hook(name, layer_info)
            )
            self._hooks.append(hook)

    def _make_hook(self, name, layer_info):
        """Create a forward hook for a specific layer."""
        import torch

        def hook_fn(module, input, output):
            ts = time.monotonic_ns() - self._start_time
            layer_info.forward_count += 1
            layer_info.timestamps_ns.append(ts)

            # Compute output activation norm
            if isinstance(output, torch.Tensor):
                with torch.no_grad():
                    norm = output.float().norm().item()
            elif isinstance(output, tuple) and len(output) > 0:
                first = output[0]
                if isinstance(first, torch.Tensor):
                    with torch.no_grad():
                        norm = first.float().norm().item()
                else:
                    norm = 0.0
            else:
                norm = 0.0

            layer_info.output_norms.append(norm)
            layer_info.total_activation += norm
            layer_info.max_activation = max(layer_info.max_activation, norm)

            # Record to access log (same format as Membrane)
            self._access_log.append((
                ts, "READ", name, layer_info.param_bytes
            ))

            # Per-head activation tracking for attention
            if layer_info.is_attention and isinstance(output, tuple):
                # Many attention implementations return (attn_output, attn_weights)
                if len(output) >= 2 and output[1] is not None:
                    attn_weights = output[1]
                    if isinstance(attn_weights, torch.Tensor):
                        with torch.no_grad():
                            # attn_weights shape: (batch, num_heads, seq, seq)
                            if attn_weights.dim() >= 2:
                                num_heads = min(attn_weights.shape[1]
                                              if attn_weights.dim() >= 3
                                              else attn_weights.shape[0],
                                              len(layer_info.head_activations)
                                              if layer_info.head_activations else 999)
                                if num_heads > 0 and not layer_info.head_activations:
                                    layer_info.head_activations = [0.0] * num_heads
                                for h in range(min(num_heads, len(layer_info.head_activations))):
                                    if attn_weights.dim() >= 3:
                                        head_norm = attn_weights[:, h].float().norm().item()
                                    else:
                                        head_norm = attn_weights[h].float().norm().item()
                                    layer_info.head_activations[h] += head_norm

        return hook_fn

    def reset(self):
        """Clear all recorded activations."""
        self._start_time = time.monotonic_ns()
        self._access_log.clear()
        for layer in self.layers.values():
            layer.forward_count = 0
            layer.total_activation = 0.0
            layer.max_activation = 0.0
            layer.output_norms.clear()
            layer.timestamps_ns.clear()
            layer.head_activations = [0.0] * len(layer.head_activations)

    def remove_hooks(self):
        """Remove all forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def to_access_log(self):
        """Return access log in Membrane-compatible format."""
        return self._access_log

    def get_activation_map(self):
        """Return layer activation summary."""
        layers = []
        for name, info in self.layers.items():
            if info.forward_count == 0:
                continue
            avg_norm = (info.total_activation / info.forward_count
                       if info.forward_count > 0 else 0)
            layers.append({
                "name": name,
                "forward_count": info.forward_count,
                "avg_activation": round(avg_norm, 4),
                "max_activation": round(info.max_activation, 4),
                "param_bytes": info.param_bytes,
                "param_mb": round(info.param_bytes / (1024 * 1024), 3),
                "is_attention": info.is_attention,
                "temperature": "HOT" if avg_norm > self.activation_threshold else "COLD",
                "head_activations": info.head_activations,
            })
        return sorted(layers, key=lambda x: -x["avg_activation"])

    def get_cold_layers(self, percentile=25):
        """Return layers below the activation percentile — candidates for condensation."""
        activation_map = self.get_activation_map()
        if not activation_map:
            return []

        activations = [l["avg_activation"] for l in activation_map]
        threshold = np.percentile(activations, percentile) if activations else 0

        return [l for l in activation_map if l["avg_activation"] <= threshold]

    def get_condensation_potential(self):
        """Calculate how much RAM could be saved by condensing cold layers."""
        activation_map = self.get_activation_map()
        if not activation_map:
            return {"total_mb": 0, "cold_mb": 0, "savings_pct": 0}

        total_bytes = sum(l["param_bytes"] for l in activation_map)
        cold_layers = self.get_cold_layers()
        cold_bytes = sum(l["param_bytes"] for l in cold_layers)

        return {
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "hot_mb": round((total_bytes - cold_bytes) / (1024 * 1024), 2),
            "cold_mb": round(cold_bytes / (1024 * 1024), 2),
            "savings_pct": round(cold_bytes / total_bytes * 100, 1) if total_bytes > 0 else 0,
            "total_layers": len(activation_map),
            "cold_layers": len(cold_layers),
            "hot_layers": len(activation_map) - len(cold_layers),
        }

    def print_activation_map(self, top_n=30):
        """Print activation summary."""
        activation_map = self.get_activation_map()
        potential = self.get_condensation_potential()

        print(f"\n{'='*70}")
        print(f"  CONDENSATE — PyTorch Activation Map")
        print(f"{'='*70}")
        print(f"  Total layers tracked: {potential['total_layers']}")
        print(f"  HOT (active):         {potential['hot_layers']} "
              f"({potential['hot_mb']:.2f} MB)")
        print(f"  COLD (condensable):   {potential['cold_layers']} "
              f"({potential['cold_mb']:.2f} MB)")
        print(f"  Potential savings:    {potential['savings_pct']:.1f}%")

        print(f"\n  {'Layer':<40} {'Fwd':>4} {'AvgAct':>8} {'MB':>6} {'Tier':>5}")
        print(f"  {'-'*40} {'-'*4} {'-'*8} {'-'*6} {'-'*5}")

        for layer in activation_map[:top_n]:
            name = layer['name']
            if len(name) > 40:
                name = "..." + name[-37:]
            tier = layer['temperature']
            attn_marker = " [A]" if layer['is_attention'] else ""
            print(f"  {name:<40} {layer['forward_count']:>4} "
                  f"{layer['avg_activation']:>8.3f} "
                  f"{layer['param_mb']:>6.3f} {tier:>5}{attn_marker}")

        if len(activation_map) > top_n:
            print(f"  ... and {len(activation_map) - top_n} more layers")

        print(f"\n{'='*70}\n")
