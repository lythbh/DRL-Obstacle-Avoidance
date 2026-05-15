"""Webots simulation stack for the ALTINO project."""

from .webots_env import (
    AltinoDriver,
    SLAMProcessor,
    WebotsEnv,
    _init_supervisor,
)

# Backward-compat aliases
MotorController = AltinoDriver
SensorReader = AltinoDriver
