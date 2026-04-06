"""Condensate Membrane — thin orchestration wrapper.

The data path is Rust. This module provides the Python API
for starting, stopping, and monitoring Condensate.
"""
import condensate_core


class Membrane:
    """Orchestration wrapper. Data path is Rust."""
    def __init__(self):
        self._active = False

    def start(self):
        """Enable membrane observation."""
        self._active = True

    def stop(self):
        """Disable membrane."""
        self._active = False

    @property
    def active(self):
        return self._active

    def status(self):
        """Return current membrane status."""
        return {"active": self._active}
