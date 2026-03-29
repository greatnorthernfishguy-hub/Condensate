"""
Condensate Layer 0: The Membrane

Intercepts and records memory access patterns on wrapped objects.
No intelligence — pure observation. Produces an access log that
Layer 1 (the graph builder) will analyze.

Usage:
    from membrane import Membrane

    data = {"weights": big_array, "config": {...}, "cache": {...}}
    wrapped = Membrane.wrap(data, name="model_state")

    # Use wrapped exactly like data — reads, writes, iteration all work
    x = wrapped["weights"]      # recorded: READ model_state.weights
    wrapped["cache"]["key"] = v  # recorded: READ model_state.cache, WRITE model_state.cache.key

    # Get the access log
    log = Membrane.get_log()     # [(timestamp_ns, event_type, path, size_bytes), ...]

    # Get stats
    Membrane.print_stats()       # Summary of access patterns
"""

import time
import sys
from collections import defaultdict


class AccessLog:
    """Central access log. All Membrane instances write here."""

    def __init__(self):
        self.entries = []
        self.start_time = time.monotonic_ns()
        self._counts = defaultdict(int)

    def record(self, event_type, path, size_bytes=0):
        """Record an access event.

        Args:
            event_type: 'READ' or 'WRITE'
            path: dotted path like 'model_state.weights.layer_0'
            size_bytes: approximate size of the accessed object
        """
        ts = time.monotonic_ns() - self.start_time
        self.entries.append((ts, event_type, path, size_bytes))
        self._counts[path] += 1

    def clear(self):
        self.entries.clear()
        self._counts.clear()
        self.start_time = time.monotonic_ns()

    def stats(self):
        """Return access statistics."""
        if not self.entries:
            return {"total_accesses": 0}

        paths = defaultdict(lambda: {"reads": 0, "writes": 0, "total_bytes": 0,
                                      "first_ns": float('inf'), "last_ns": 0})

        for ts, event_type, path, size_bytes in self.entries:
            p = paths[path]
            if event_type == "READ":
                p["reads"] += 1
            else:
                p["writes"] += 1
            p["total_bytes"] += size_bytes
            p["first_ns"] = min(p["first_ns"], ts)
            p["last_ns"] = max(p["last_ns"], ts)

        # Find temporal co-access: paths accessed within window of each other
        window_ns = 1_000_000  # 1ms window
        coaccesses = defaultdict(int)
        sorted_entries = sorted(self.entries, key=lambda e: e[0])

        for i, (ts_i, _, path_i, _) in enumerate(sorted_entries):
            for j in range(i + 1, len(sorted_entries)):
                ts_j, _, path_j, _ = sorted_entries[j]
                if ts_j - ts_i > window_ns:
                    break
                if path_i != path_j:
                    pair = tuple(sorted([path_i, path_j]))
                    coaccesses[pair] += 1

        duration_ms = (self.entries[-1][0] - self.entries[0][0]) / 1_000_000

        return {
            "total_accesses": len(self.entries),
            "unique_paths": len(paths),
            "duration_ms": round(duration_ms, 2),
            "paths": dict(paths),
            "top_coaccesses": sorted(coaccesses.items(),
                                      key=lambda x: -x[1])[:20],
        }

    def print_stats(self):
        """Print a readable summary."""
        s = self.stats()
        print(f"\n{'='*60}")
        print(f"  CONDENSATE MEMBRANE — Access Log Summary")
        print(f"{'='*60}")
        print(f"  Total accesses:  {s['total_accesses']}")
        print(f"  Unique paths:    {s['unique_paths']}")
        print(f"  Duration:        {s['duration_ms']} ms")

        if s.get("paths"):
            print(f"\n  {'Path':<40} {'Reads':>6} {'Writes':>6}")
            print(f"  {'-'*40} {'-'*6} {'-'*6}")

            # Sort by total access count
            sorted_paths = sorted(s["paths"].items(),
                                   key=lambda x: -(x[1]["reads"] + x[1]["writes"]))

            for path, info in sorted_paths[:25]:
                # Truncate long paths
                display = path if len(path) <= 40 else "..." + path[-37:]
                print(f"  {display:<40} {info['reads']:>6} {info['writes']:>6}")

            if len(sorted_paths) > 25:
                print(f"  ... and {len(sorted_paths) - 25} more paths")

        if s.get("top_coaccesses"):
            print(f"\n  Top co-accesses (within 1ms window):")
            print(f"  {'-'*54}")
            for (a, b), count in s["top_coaccesses"][:10]:
                a_short = a if len(a) <= 22 else "..." + a[-19:]
                b_short = b if len(b) <= 22 else "..." + b[-19:]
                print(f"  {a_short:<22} <-> {b_short:<22} {count:>4}x")

        print(f"{'='*60}\n")


# Global singleton log
_log = AccessLog()


def _obj_size(obj):
    """Rough size estimate without deep traversal."""
    try:
        return sys.getsizeof(obj)
    except (TypeError, AttributeError):
        return 0


class MembraneDict(dict):
    """A dict wrapper that records access patterns."""

    def __init__(self, data, path, log):
        super().__init__(data)
        self._membrane_path = path
        self._membrane_log = log

    def __getitem__(self, key):
        full_path = f"{self._membrane_path}.{key}"
        value = super().__getitem__(key)
        self._membrane_log.record("READ", full_path, _obj_size(value))

        # Wrap nested containers so we track deep access
        if isinstance(value, dict) and not isinstance(value, MembraneDict):
            wrapped = MembraneDict(value, full_path, self._membrane_log)
            super().__setitem__(key, wrapped)
            return wrapped
        if isinstance(value, list) and not isinstance(value, MembraneList):
            wrapped = MembraneList(value, full_path, self._membrane_log)
            super().__setitem__(key, wrapped)
            return wrapped

        return value

    def __setitem__(self, key, value):
        full_path = f"{self._membrane_path}.{key}"
        self._membrane_log.record("WRITE", full_path, _obj_size(value))
        super().__setitem__(key, value)

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __repr__(self):
        return f"MembraneDict({self._membrane_path}, {len(self)} keys)"


class MembraneList(list):
    """A list wrapper that records access patterns."""

    def __init__(self, data, path, log):
        super().__init__(data)
        self._membrane_path = path
        self._membrane_log = log

    def __getitem__(self, index):
        full_path = f"{self._membrane_path}[{index}]"
        value = super().__getitem__(index)
        self._membrane_log.record("READ", full_path, _obj_size(value))

        if isinstance(value, dict) and not isinstance(value, MembraneDict):
            wrapped = MembraneDict(value, full_path, self._membrane_log)
            super().__setitem__(index, wrapped)
            return wrapped

        return value

    def __setitem__(self, index, value):
        full_path = f"{self._membrane_path}[{index}]"
        self._membrane_log.record("WRITE", full_path, _obj_size(value))
        super().__setitem__(index, value)

    def __repr__(self):
        return f"MembraneList({self._membrane_path}, {len(self)} items)"


class MembraneObject:
    """Wraps an arbitrary Python object to record attribute access."""

    def __init__(self, obj, path, log):
        object.__setattr__(self, '_membrane_obj', obj)
        object.__setattr__(self, '_membrane_path', path)
        object.__setattr__(self, '_membrane_log', log)

    def __getattr__(self, name):
        if name.startswith('_membrane_'):
            return object.__getattribute__(self, name)

        obj = object.__getattribute__(self, '_membrane_obj')
        path = object.__getattribute__(self, '_membrane_path')
        log = object.__getattribute__(self, '_membrane_log')

        full_path = f"{path}.{name}"
        value = getattr(obj, name)
        log.record("READ", full_path, _obj_size(value))

        # Wrap nested containers
        if isinstance(value, dict) and not isinstance(value, MembraneDict):
            return MembraneDict(value, full_path, log)
        if isinstance(value, list) and not isinstance(value, MembraneList):
            return MembraneList(value, full_path, log)

        return value

    def __setattr__(self, name, value):
        if name.startswith('_membrane_'):
            object.__setattr__(self, name, value)
            return

        obj = object.__getattribute__(self, '_membrane_obj')
        path = object.__getattribute__(self, '_membrane_path')
        log = object.__getattribute__(self, '_membrane_log')

        full_path = f"{path}.{name}"
        log.record("WRITE", full_path, _obj_size(value))
        setattr(obj, name, value)

    def __repr__(self):
        obj = object.__getattribute__(self, '_membrane_obj')
        path = object.__getattribute__(self, '_membrane_path')
        return f"MembraneObject({path}, {type(obj).__name__})"


class Membrane:
    """Factory for wrapping objects with access tracking.

    Example:
        data = {"a": [1, 2, 3], "b": {"nested": True}}
        wrapped = Membrane.wrap(data, "my_data")
        x = wrapped["a"]       # logged
        y = wrapped["b"]["nested"]  # both accesses logged
        Membrane.print_stats()
    """

    @staticmethod
    def wrap(obj, name="root"):
        """Wrap an object for access tracking.

        Args:
            obj: Any Python object (dict, list, or arbitrary object)
            name: Human-readable name for this object in the log
        """
        if isinstance(obj, dict):
            return MembraneDict(obj, name, _log)
        elif isinstance(obj, list):
            return MembraneList(obj, name, _log)
        else:
            return MembraneObject(obj, name, _log)

    @staticmethod
    def get_log():
        """Get the raw access log entries."""
        return _log.entries

    @staticmethod
    def stats():
        """Get access statistics as a dict."""
        return _log.stats()

    @staticmethod
    def print_stats():
        """Print a readable summary of access patterns."""
        _log.print_stats()

    @staticmethod
    def clear():
        """Clear the access log."""
        _log.clear()

    @staticmethod
    def entry_count():
        """Quick check: how many accesses recorded."""
        return len(_log.entries)

    @staticmethod
    def save_log(filepath):
        """Save the raw log to a file for Layer 1 analysis."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "entries": _log.entries,
                "stats": {
                    "total": len(_log.entries),
                    "unique_paths": len(set(e[2] for e in _log.entries)),
                }
            }, f, indent=2)
        print(f"  Saved {len(_log.entries)} entries to {filepath}")
