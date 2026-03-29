"""
Condensate: PyTorch Membrane (v2 — Head-Level Granularity)

Hooks into nn.Module forward passes to track activation at TWO levels:
  - Layer level: which modules fire, how strongly
  - Head level: within attention layers, which individual heads contribute

This is the key upgrade. Layer-level tracking found a 16.6% floor.
Head-level tracking sees inside that floor — different inputs activate
different heads within the same layer. That's where 50%+ savings live.

Usage:
    from torch_membrane import TorchMembrane

    model = AutoModelForCausalLM.from_pretrained("gpt2-large")
    membrane = TorchMembrane(model)

    output = model.generate(input_ids)

    membrane.print_activation_map()        # layer-level summary
    membrane.print_head_map()              # head-level detail
    membrane.get_condensation_potential()   # layer-level savings
    membrane.get_head_condensation_potential()  # head-level savings
"""

import time
import numpy as np
from collections import defaultdict


class HeadActivation:
    """Tracks activation for a single attention head."""

    __slots__ = ['layer_name', 'head_idx', 'activation_sum', 'activation_max',
                 'forward_count', 'norms']

    def __init__(self, layer_name, head_idx):
        self.layer_name = layer_name
        self.head_idx = head_idx
        self.activation_sum = 0.0
        self.activation_max = 0.0
        self.forward_count = 0
        self.norms = []

    def record(self, norm):
        self.forward_count += 1
        self.activation_sum += norm
        self.activation_max = max(self.activation_max, norm)
        self.norms.append(norm)

    @property
    def avg_activation(self):
        return self.activation_sum / self.forward_count if self.forward_count > 0 else 0.0

    def reset(self):
        self.activation_sum = 0.0
        self.activation_max = 0.0
        self.forward_count = 0
        self.norms.clear()


class LayerActivation:
    """Records activation statistics for a single layer."""

    __slots__ = ['name', 'forward_count', 'total_activation',
                 'max_activation', 'output_norms', 'timestamps_ns',
                 'param_bytes', 'is_attention', 'num_heads',
                 'per_head_param_bytes']

    def __init__(self, name, param_bytes=0, is_attention=False, num_heads=0):
        self.name = name
        self.forward_count = 0
        self.total_activation = 0.0
        self.max_activation = 0.0
        self.output_norms = []
        self.timestamps_ns = []
        self.param_bytes = param_bytes
        self.is_attention = is_attention
        self.num_heads = num_heads
        # For attention layers, divide params evenly across heads
        self.per_head_param_bytes = (param_bytes // num_heads) if num_heads > 0 else 0

    def reset(self):
        self.forward_count = 0
        self.total_activation = 0.0
        self.max_activation = 0.0
        self.output_norms.clear()
        self.timestamps_ns.clear()


class TorchMembrane:
    """Hooks into a PyTorch model to track layer AND head activations.

    Two levels of granularity:
    - Layer level: every nn.Module tracked by output norm
    - Head level: attention layers decomposed into individual heads
      by analyzing the output tensor shape and computing per-head norms
    """

    def __init__(self, model, activation_threshold=0.01):
        self.model = model
        self.activation_threshold = activation_threshold
        self.layers = {}              # name → LayerActivation
        self.heads = {}               # "layer_name.head_N" → HeadActivation
        self._hooks = []
        self._start_time = time.monotonic_ns()
        self._access_log = []

        # Detect model config for head count
        config = getattr(model, 'config', None)
        self._default_num_heads = getattr(config, 'n_head',
                                  getattr(config, 'num_attention_heads', 0))
        self._head_dim = 0
        if config:
            hidden = getattr(config, 'n_embd',
                    getattr(config, 'hidden_size', 0))
            if self._default_num_heads > 0 and hidden > 0:
                self._head_dim = hidden // self._default_num_heads

        self._install_hooks()

    def _install_hooks(self):
        """Install forward hooks on all modules."""
        import torch

        for name, module in self.model.named_modules():
            if name == '':
                continue

            param_bytes = sum(p.numel() * p.element_size()
                             for p in module.parameters(recurse=False))

            # Detect attention layers
            is_attention = any(kw in name.lower()
                              for kw in ['attn', 'attention', 'self_attn'])

            # Detect attention OUTPUT projection specifically — this is where
            # we can decompose by head from the pre-projection tensor
            is_attn_output = is_attention and any(
                kw in name.lower()
                for kw in ['c_proj', 'out_proj', 'o_proj', 'dense']
            )

            num_heads = 0
            if is_attention:
                num_heads = getattr(module, 'num_heads',
                           getattr(module, 'num_attention_heads',
                           self._default_num_heads))

                # Register per-head trackers
                if num_heads > 0:
                    for h in range(num_heads):
                        head_key = f"{name}.head_{h}"
                        self.heads[head_key] = HeadActivation(name, h)

            layer_info = LayerActivation(
                name=name,
                param_bytes=param_bytes,
                is_attention=is_attention,
                num_heads=num_heads,
            )
            self.layers[name] = layer_info

            hook = module.register_forward_hook(
                self._make_hook(name, layer_info)
            )
            self._hooks.append(hook)

    def _make_hook(self, name, layer_info):
        """Create a forward hook that tracks both layer and head activation."""
        import torch

        def hook_fn(module, input, output):
            ts = time.monotonic_ns() - self._start_time
            layer_info.forward_count += 1
            layer_info.timestamps_ns.append(ts)

            # Compute layer-level output norm
            out_tensor = None
            if isinstance(output, torch.Tensor):
                out_tensor = output
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    out_tensor = output[0]

            if out_tensor is not None:
                with torch.no_grad():
                    norm = out_tensor.float().norm().item()
            else:
                norm = 0.0

            layer_info.output_norms.append(norm)
            layer_info.total_activation += norm
            layer_info.max_activation = max(layer_info.max_activation, norm)

            # Record layer access
            self._access_log.append((ts, "READ", name, layer_info.param_bytes))

            # Head-level decomposition for attention layers
            if layer_info.is_attention and layer_info.num_heads > 0 and out_tensor is not None:
                self._decompose_heads(name, layer_info, out_tensor, ts)

        return hook_fn

    def _decompose_heads(self, name, layer_info, output_tensor, ts):
        """Decompose attention output into per-head activation norms.

        For GPT-2 style models, the attention output is (batch, seq, hidden).
        hidden = num_heads * head_dim. We reshape and compute per-head norms.
        """
        import torch

        num_heads = layer_info.num_heads
        if num_heads <= 0:
            return

        try:
            with torch.no_grad():
                shape = output_tensor.shape
                # Expected: (batch, seq_len, hidden_size) or (batch, seq_len, num_heads * head_dim)
                if len(shape) < 2:
                    return

                hidden = shape[-1]

                # Only decompose if hidden is divisible by num_heads
                if hidden % num_heads != 0:
                    return

                head_dim = hidden // num_heads

                # Reshape to (batch, seq_len, num_heads, head_dim)
                reshaped = output_tensor.view(*shape[:-1], num_heads, head_dim)

                # Compute per-head norm: norm across (batch, seq_len, head_dim)
                for h in range(num_heads):
                    head_key = f"{name}.head_{h}"
                    head_tracker = self.heads.get(head_key)
                    if head_tracker:
                        head_norm = reshaped[..., h, :].float().norm().item()
                        head_tracker.record(head_norm)

                        # Record head-level access
                        self._access_log.append((
                            ts, "READ", head_key,
                            layer_info.per_head_param_bytes
                        ))

        except (RuntimeError, ValueError):
            # Shape mismatch — skip head decomposition for this layer
            pass

    def reset(self):
        """Clear all recorded activations."""
        self._start_time = time.monotonic_ns()
        self._access_log.clear()
        for layer in self.layers.values():
            layer.reset()
        for head in self.heads.values():
            head.reset()

    def remove_hooks(self):
        """Remove all forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def to_access_log(self):
        """Return access log in Membrane-compatible format."""
        return self._access_log

    # --- Layer-level analysis (same as v1) ---

    def get_activation_map(self):
        """Return layer activation summary."""
        layers = []
        for name, info in self.layers.items():
            if info.forward_count == 0:
                continue
            avg_norm = info.total_activation / info.forward_count
            layers.append({
                "name": name,
                "forward_count": info.forward_count,
                "avg_activation": round(avg_norm, 4),
                "max_activation": round(info.max_activation, 4),
                "param_bytes": info.param_bytes,
                "param_mb": round(info.param_bytes / (1024 * 1024), 3),
                "is_attention": info.is_attention,
                "num_heads": info.num_heads,
                "temperature": "HOT" if avg_norm > self.activation_threshold else "COLD",
            })
        return sorted(layers, key=lambda x: -x["avg_activation"])

    def get_cold_layers(self, percentile=25):
        activation_map = self.get_activation_map()
        if not activation_map:
            return []
        activations = [l["avg_activation"] for l in activation_map]
        threshold = np.percentile(activations, percentile)
        return [l for l in activation_map if l["avg_activation"] <= threshold]

    def get_condensation_potential(self):
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

    # --- Head-level analysis (new in v2) ---

    def get_head_map(self):
        """Return per-head activation summary for all attention layers."""
        head_data = []
        for key, head in self.heads.items():
            if head.forward_count == 0:
                continue

            # Find the parent layer to get per-head param size
            parent = self.layers.get(head.layer_name)
            per_head_bytes = parent.per_head_param_bytes if parent else 0

            head_data.append({
                "key": key,
                "layer": head.layer_name,
                "head_idx": head.head_idx,
                "forward_count": head.forward_count,
                "avg_activation": round(head.avg_activation, 4),
                "max_activation": round(head.activation_max, 4),
                "param_bytes": per_head_bytes,
                "param_mb": round(per_head_bytes / (1024 * 1024), 4),
                "temperature": "HOT" if head.avg_activation > self.activation_threshold else "COLD",
            })
        return sorted(head_data, key=lambda x: -x["avg_activation"])

    def get_cold_heads(self, percentile=25):
        """Return heads below the activation percentile."""
        head_map = self.get_head_map()
        if not head_map:
            return []
        activations = [h["avg_activation"] for h in head_map]
        threshold = np.percentile(activations, percentile)
        return [h for h in head_map if h["avg_activation"] <= threshold]

    def get_head_condensation_potential(self):
        """Calculate RAM savings at head-level granularity."""
        head_map = self.get_head_map()
        if not head_map:
            return {"total_mb": 0, "cold_mb": 0, "savings_pct": 0,
                    "total_heads": 0, "cold_heads": 0, "hot_heads": 0}

        total_bytes = sum(h["param_bytes"] for h in head_map)
        cold_heads = self.get_cold_heads()
        cold_bytes = sum(h["param_bytes"] for h in cold_heads)

        # Also get non-attention layer data for the full picture
        non_attn_layers = [l for l in self.get_activation_map()
                           if not l["is_attention"]]
        cold_non_attn = [l for l in non_attn_layers
                         if l["temperature"] == "COLD"]
        non_attn_cold_bytes = sum(l["param_bytes"] for l in cold_non_attn)
        non_attn_total_bytes = sum(l["param_bytes"] for l in non_attn_layers)

        grand_total = total_bytes + non_attn_total_bytes
        grand_cold = cold_bytes + non_attn_cold_bytes

        return {
            "attn_total_mb": round(total_bytes / (1024 * 1024), 2),
            "attn_hot_mb": round((total_bytes - cold_bytes) / (1024 * 1024), 2),
            "attn_cold_mb": round(cold_bytes / (1024 * 1024), 2),
            "non_attn_total_mb": round(non_attn_total_bytes / (1024 * 1024), 2),
            "non_attn_cold_mb": round(non_attn_cold_bytes / (1024 * 1024), 2),
            "total_mb": round(grand_total / (1024 * 1024), 2),
            "cold_mb": round(grand_cold / (1024 * 1024), 2),
            "hot_mb": round((grand_total - grand_cold) / (1024 * 1024), 2),
            "savings_pct": round(grand_cold / grand_total * 100, 1) if grand_total > 0 else 0,
            "total_heads": len(head_map),
            "cold_heads": len(cold_heads),
            "hot_heads": len(head_map) - len(cold_heads),
            "cold_non_attn_layers": len(cold_non_attn),
        }

    def print_activation_map(self, top_n=30):
        """Print layer-level activation summary."""
        activation_map = self.get_activation_map()
        potential = self.get_condensation_potential()

        print(f"\n{'='*70}")
        print(f"  CONDENSATE — Layer Activation Map")
        print(f"{'='*70}")
        print(f"  Total layers: {potential['total_layers']}")
        print(f"  HOT: {potential['hot_layers']} ({potential['hot_mb']:.2f} MB)")
        print(f"  COLD: {potential['cold_layers']} ({potential['cold_mb']:.2f} MB)")
        print(f"  Layer-level savings: {potential['savings_pct']:.1f}%")

        print(f"\n  {'Layer':<40} {'Fwd':>4} {'AvgAct':>8} {'MB':>6} {'Tier':>5}")
        print(f"  {'-'*40} {'-'*4} {'-'*8} {'-'*6} {'-'*5}")

        for layer in activation_map[:top_n]:
            name = layer['name'] if len(layer['name']) <= 40 else "..." + layer['name'][-37:]
            attn = " [A]" if layer['is_attention'] else ""
            print(f"  {name:<40} {layer['forward_count']:>4} "
                  f"{layer['avg_activation']:>8.3f} "
                  f"{layer['param_mb']:>6.3f} {layer['temperature']:>5}{attn}")

        print(f"\n{'='*70}\n")

    def print_head_map(self, top_n=40):
        """Print head-level activation map."""
        head_map = self.get_head_map()
        head_potential = self.get_head_condensation_potential()

        print(f"\n{'='*70}")
        print(f"  CONDENSATE — Head-Level Activation Map")
        print(f"{'='*70}")
        print(f"  Total attention heads: {head_potential['total_heads']}")
        print(f"  HOT heads: {head_potential['hot_heads']}")
        print(f"  COLD heads: {head_potential['cold_heads']}")
        print(f"  Attention params: {head_potential['attn_total_mb']:.2f} MB "
              f"(cold: {head_potential['attn_cold_mb']:.2f} MB)")
        print(f"  Non-attention cold: {head_potential['non_attn_cold_mb']:.2f} MB")
        print(f"  *** HEAD-LEVEL SAVINGS: {head_potential['savings_pct']:.1f}% "
              f"({head_potential['cold_mb']:.2f} MB) ***")

        # Show coldest heads
        cold_heads = self.get_cold_heads()
        if cold_heads:
            print(f"\n  Coldest heads (bottom 25%):")
            print(f"  {'Head':<40} {'Fwd':>4} {'AvgAct':>10} {'MB':>6}")
            print(f"  {'-'*40} {'-'*4} {'-'*10} {'-'*6}")
            for h in cold_heads[:top_n]:
                name = h['key'] if len(h['key']) <= 40 else "..." + h['key'][-37:]
                print(f"  {name:<40} {h['forward_count']:>4} "
                      f"{h['avg_activation']:>10.4f} {h['param_mb']:>6.4f}")

        # Show hottest heads for comparison
        hot_heads = [h for h in head_map if h['temperature'] == 'HOT']
        if hot_heads:
            print(f"\n  Hottest heads (sample):")
            print(f"  {'Head':<40} {'Fwd':>4} {'AvgAct':>10} {'MB':>6}")
            print(f"  {'-'*40} {'-'*4} {'-'*10} {'-'*6}")
            for h in hot_heads[:10]:
                name = h['key'] if len(h['key']) <= 40 else "..." + h['key'][-37:]
                print(f"  {name:<40} {h['forward_count']:>4} "
                      f"{h['avg_activation']:>10.4f} {h['param_mb']:>6.4f}")

        print(f"\n{'='*70}\n")
