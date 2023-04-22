"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: test
import pytest

# IMPORT: project
from src.training.components.dashboard import Dashboard, \
    Dashboard2D, Dashboard25D, Dashboard3D


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def dashboard():
    return Dashboard({"duration": 5}, train_id="test")


@pytest.fixture(scope="function")
def dashboard_2d():
    return Dashboard2D({"duration": 5}, train_id="test")


@pytest.fixture(scope="function")
def dashboard_25d():
    return Dashboard25D({"duration": 5}, train_id="test")


@pytest.fixture(scope="function")
def dashboard_3d():
    return Dashboard3D({"duration": 5}, train_id="test")


# -------------------- DASHBOARD -------------------- #

def test_dashboard(dashboard):
    pass
