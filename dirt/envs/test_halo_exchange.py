#!/usr/bin/env python3
"""
Unit tests for food halo exchange in distributed environment.

Tests verify that:
1. Halo regions are correctly populated with neighbor data
2. Interior regions remain unchanged during exchange
3. Boundary devices handle edges correctly
4. Shape and dtype are preserved
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import pmap

from dirt.envs.simple_env_dist import make_food_halo_copy

AXIS = "mesh"
TOLERANCE = 1e-6  # Numerical comparison tolerance


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def grid_config():
    """
    Standard grid configuration for tests.

    Returns:
        dict: Configuration with keys 'H' (height), 'W' (width), 'halo' (size)
    """
    return {
        'H': 8,
        'W': 8,
        'halo': 2,
    }


@pytest.fixture
def device_setup(grid_config):
    """
    Setup JAX devices and create pmapped halo copy function.

    Args:
        grid_config: Grid configuration fixture

    Returns:
        dict: Setup containing 'ndev', 'devices', 'halo_copy_pm'
    """
    ndev = min(4, jax.device_count())
    devices = jax.devices()[:ndev]

    halo_copy = make_food_halo_copy(
        grid_config['H'],
        grid_config['W'],
        grid_config['halo'],
        ndev
    )
    halo_copy_pm = pmap(halo_copy, in_axes=(0, 0), axis_name=AXIS, devices=devices)

    return {
        'ndev': ndev,
        'devices': devices,
        'halo_copy_pm': halo_copy_pm,
    }


@pytest.fixture
def constant_test_grids(grid_config, device_setup):
    """
    Create test grids with device-specific constant values.

    Each device has:
    - Interior: constant value = dev_id + 1
    - Halo: constant value = -(dev_id + 1)

    This pattern makes it easy to verify correct data exchange.

    Args:
        grid_config: Grid configuration fixture
        device_setup: Device setup fixture

    Returns:
        jnp.ndarray: Shape (ndev, H, W + 2*halo)
    """
    H = grid_config['H']
    W = grid_config['W']
    halo = grid_config['halo']
    ndev = device_setup['ndev']

    grids = []
    for dev_id in range(ndev):
        # Interior value is positive, halo is negative
        interior_value = float(dev_id + 1)
        halo_value = -float(dev_id + 1)

        # Create full grid with halo value
        grid = jnp.full((H, W + 2*halo), halo_value, dtype=jnp.float32)
        # Set interior to interior value
        grid = grid.at[:, halo:halo+W].set(interior_value)
        grids.append(grid)

    return jnp.stack(grids, axis=0)


@pytest.fixture
def gradient_test_grids(grid_config, device_setup):
    """
    Create test grids with gradient patterns for more realistic testing.

    Each device has a unique gradient pattern in the interior that can be
    verified after exchange.

    Args:
        grid_config: Grid configuration fixture
        device_setup: Device setup fixture

    Returns:
        jnp.ndarray: Shape (ndev, H, W + 2*halo)
    """
    H = grid_config['H']
    W = grid_config['W']
    halo = grid_config['halo']
    ndev = device_setup['ndev']

    grids = []
    for dev_id in range(ndev):
        # Create gradient: each column has value = dev_id * 100 + col_idx
        grid = jnp.zeros((H, W + 2*halo), dtype=jnp.float32)
        for col in range(W):
            value = float(dev_id * 100 + col)
            grid = grid.at[:, halo + col].set(value)
        grids.append(grid)

    return jnp.stack(grids, axis=0)


# ============================================================================
# Test Cases
# ============================================================================

class TestHaloExchangeBasic:
    """Basic halo exchange functionality tests."""

    def test_output_shape_preserved(self, grid_config, device_setup, constant_test_grids):
        """Verify output shape matches input shape."""
        halo_copy_pm = device_setup['halo_copy_pm']
        ndev = device_setup['ndev']
        roles = jnp.arange(ndev, dtype=jnp.int32)

        result = halo_copy_pm(constant_test_grids, roles)

        assert result.shape == constant_test_grids.shape, \
            f"Output shape {result.shape} does not match input shape {constant_test_grids.shape}"

    def test_output_dtype_preserved(self, grid_config, device_setup, constant_test_grids):
        """Verify output dtype matches input dtype."""
        halo_copy_pm = device_setup['halo_copy_pm']
        ndev = device_setup['ndev']
        roles = jnp.arange(ndev, dtype=jnp.int32)

        result = halo_copy_pm(constant_test_grids, roles)

        assert result.dtype == constant_test_grids.dtype, \
            f"Output dtype {result.dtype} does not match input dtype {constant_test_grids.dtype}"

    def test_interior_unchanged(self, grid_config, device_setup, constant_test_grids):
        """Verify interior regions remain unchanged after halo exchange."""
        H = grid_config['H']
        W = grid_config['W']
        halo = grid_config['halo']
        ndev = device_setup['ndev']
        halo_copy_pm = device_setup['halo_copy_pm']

        L = halo
        R = halo + W
        roles = jnp.arange(ndev, dtype=jnp.int32)

        # Get original interior values
        original_interiors = constant_test_grids[:, :, L:R]

        # Perform halo exchange
        result = halo_copy_pm(constant_test_grids, roles)
        result_cpu = jax.device_get(result)
        result_interiors = result_cpu[:, :, L:R]

        # Verify all interiors unchanged
        assert np.allclose(result_interiors, original_interiors, atol=TOLERANCE), \
            "Interior regions were modified during halo exchange"


class TestHaloExchangeData:
    """Tests for correct data exchange between devices."""

    def test_left_halo_receives_neighbor_data(self, grid_config, device_setup, constant_test_grids):
        """Verify left halo receives data from left neighbor's right border."""
        H = grid_config['H']
        W = grid_config['W']
        halo = grid_config['halo']
        ndev = device_setup['ndev']
        halo_copy_pm = device_setup['halo_copy_pm']

        L = halo
        roles = jnp.arange(ndev, dtype=jnp.int32)

        result = halo_copy_pm(constant_test_grids, roles)
        result_cpu = jax.device_get(result)

        # Check all non-leftmost devices
        for dev_id in range(1, ndev):
            left_halo = result_cpu[dev_id, :, 0:L]
            # Left neighbor's interior value (dev_id is current, so left neighbor is dev_id - 1)
            expected_value = float(dev_id)  # Left neighbor has value dev_id (since we use dev_id + 1)

            assert np.allclose(left_halo, expected_value, atol=TOLERANCE), \
                f"Device {dev_id}: Left halo does not contain correct neighbor data. " \
                f"Expected {expected_value}, got {left_halo[0, 0]}"

    def test_right_halo_receives_neighbor_data(self, grid_config, device_setup, constant_test_grids):
        """Verify right halo receives data from right neighbor's left border."""
        H = grid_config['H']
        W = grid_config['W']
        halo = grid_config['halo']
        ndev = device_setup['ndev']
        halo_copy_pm = device_setup['halo_copy_pm']

        L = halo
        R = halo + W
        roles = jnp.arange(ndev, dtype=jnp.int32)

        result = halo_copy_pm(constant_test_grids, roles)
        result_cpu = jax.device_get(result)

        # Check all non-rightmost devices
        for dev_id in range(ndev - 1):
            right_halo = result_cpu[dev_id, :, R:R+halo]
            # Right neighbor's interior value
            expected_value = float(dev_id + 2)  # Right neighbor is dev_id + 1, which has value dev_id + 2

            assert np.allclose(right_halo, expected_value, atol=TOLERANCE), \
                f"Device {dev_id}: Right halo does not contain correct neighbor data. " \
                f"Expected {expected_value}, got {right_halo[0, 0]}"

    def test_leftmost_device_boundary(self, grid_config, device_setup, constant_test_grids):
        """Verify leftmost device's left halo remains unchanged (no left neighbor)."""
        halo = grid_config['halo']
        halo_copy_pm = device_setup['halo_copy_pm']
        ndev = device_setup['ndev']

        L = halo
        roles = jnp.arange(ndev, dtype=jnp.int32)

        # Get original left halo of device 0
        original_left_halo = constant_test_grids[0, :, 0:L]

        result = halo_copy_pm(constant_test_grids, roles)
        result_cpu = jax.device_get(result)
        result_left_halo = result_cpu[0, :, 0:L]

        expected_value = -1.0  # Original halo value for device 0

        assert np.allclose(result_left_halo, expected_value, atol=TOLERANCE), \
            f"Device 0: Left boundary halo should remain unchanged at {expected_value}"

    def test_rightmost_device_boundary(self, grid_config, device_setup, constant_test_grids):
        """Verify rightmost device's right halo remains unchanged (no right neighbor)."""
        W = grid_config['W']
        halo = grid_config['halo']
        halo_copy_pm = device_setup['halo_copy_pm']
        ndev = device_setup['ndev']

        R = halo + W
        roles = jnp.arange(ndev, dtype=jnp.int32)

        result = halo_copy_pm(constant_test_grids, roles)
        result_cpu = jax.device_get(result)
        result_right_halo = result_cpu[ndev - 1, :, R:R+halo]

        expected_value = -float(ndev)  # Original halo value for last device

        assert np.allclose(result_right_halo, expected_value, atol=TOLERANCE), \
            f"Device {ndev - 1}: Right boundary halo should remain unchanged at {expected_value}"


class TestHaloExchangeGradient:
    """Tests using gradient patterns for more realistic data."""

    def test_gradient_exchange_left(self, grid_config, device_setup, gradient_test_grids):
        """Verify gradient data is correctly exchanged to left halo."""
        W = grid_config['W']
        halo = grid_config['halo']
        ndev = device_setup['ndev']
        halo_copy_pm = device_setup['halo_copy_pm']

        L = halo
        roles = jnp.arange(ndev, dtype=jnp.int32)

        result = halo_copy_pm(gradient_test_grids, roles)
        result_cpu = jax.device_get(result)

        # Check non-leftmost devices
        for dev_id in range(1, ndev):
            left_halo = result_cpu[dev_id, :, 0:L]

            # Left neighbor's rightmost columns
            left_neighbor_id = dev_id - 1
            expected_values = []
            for col_offset in range(halo):
                col_idx = W - halo + col_offset
                expected_values.append(float(left_neighbor_id * 100 + col_idx))

            for col_offset in range(halo):
                halo_col = left_halo[:, col_offset]
                expected_value = expected_values[col_offset]

                assert np.allclose(halo_col, expected_value, atol=TOLERANCE), \
                    f"Device {dev_id}: Left halo column {col_offset} incorrect. " \
                    f"Expected {expected_value}, got {halo_col[0]}"

    def test_gradient_exchange_right(self, grid_config, device_setup, gradient_test_grids):
        """Verify gradient data is correctly exchanged to right halo."""
        W = grid_config['W']
        halo = grid_config['halo']
        ndev = device_setup['ndev']
        halo_copy_pm = device_setup['halo_copy_pm']

        L = halo
        R = halo + W
        roles = jnp.arange(ndev, dtype=jnp.int32)

        result = halo_copy_pm(gradient_test_grids, roles)
        result_cpu = jax.device_get(result)

        # Check non-rightmost devices
        for dev_id in range(ndev - 1):
            right_halo = result_cpu[dev_id, :, R:R+halo]

            # Right neighbor's leftmost columns
            right_neighbor_id = dev_id + 1
            expected_values = []
            for col_offset in range(halo):
                expected_values.append(float(right_neighbor_id * 100 + col_offset))

            for col_offset in range(halo):
                halo_col = right_halo[:, col_offset]
                expected_value = expected_values[col_offset]

                assert np.allclose(halo_col, expected_value, atol=TOLERANCE), \
                    f"Device {dev_id}: Right halo column {col_offset} incorrect. " \
                    f"Expected {expected_value}, got {halo_col[0]}"


class TestHaloExchangeEdgeCases:
    """Tests for edge cases and special configurations."""

    @pytest.mark.skipif(jax.device_count() < 1, reason="Requires at least 1 device")
    def test_single_device_no_exchange(self, grid_config):
        """Verify single device case works correctly (no actual exchange)."""
        H = grid_config['H']
        W = grid_config['W']
        halo = grid_config['halo']
        ndev = 1

        halo_copy = make_food_halo_copy(H, W, halo, ndev)
        halo_copy_pm = pmap(halo_copy, in_axes=(0, 0), axis_name=AXIS)

        # Create test grid
        grid = jnp.full((1, H, W + 2*halo), 5.0, dtype=jnp.float32)
        # Set interior
        grid = grid.at[:, :, halo:halo+W].set(10.0)

        roles = jnp.array([0], dtype=jnp.int32)
        result = halo_copy_pm(grid, roles)
        result_cpu = jax.device_get(result)

        # With single device, grid should remain unchanged
        assert np.allclose(result_cpu, grid, atol=TOLERANCE), \
            "Single device: grid should remain unchanged"

    @pytest.mark.parametrize("halo_size", [1, 2, 3, 5])
    def test_different_halo_sizes(self, halo_size):
        """Test halo exchange with different halo sizes."""
        if jax.device_count() < 2:
            pytest.skip("Requires at least 2 devices")

        H, W = 8, 8
        ndev = min(2, jax.device_count())

        halo_copy = make_food_halo_copy(H, W, halo_size, ndev)
        devices = jax.devices()[:ndev]
        halo_copy_pm = pmap(halo_copy, in_axes=(0, 0), axis_name=AXIS, devices=devices)

        # Create test grids
        grids = []
        for dev_id in range(ndev):
            grid = jnp.zeros((H, W + 2*halo_size), dtype=jnp.float32)
            grid = grid.at[:, halo_size:halo_size+W].set(float(dev_id + 1))
            grids.append(grid)
        grids = jnp.stack(grids, axis=0)

        roles = jnp.arange(ndev, dtype=jnp.int32)
        result = halo_copy_pm(grids, roles)
        result_cpu = jax.device_get(result)

        # Verify interior unchanged
        L = halo_size
        R = halo_size + W
        for dev_id in range(ndev):
            interior = result_cpu[dev_id, :, L:R]
            assert np.allclose(interior, float(dev_id + 1), atol=TOLERANCE), \
                f"Halo size {halo_size}, device {dev_id}: Interior changed"

        # Verify left halo for device 1 (if exists)
        if ndev > 1:
            left_halo = result_cpu[1, :, 0:halo_size]
            assert np.allclose(left_halo, 1.0, atol=TOLERANCE), \
                f"Halo size {halo_size}: Left halo exchange failed"


class TestHaloExchangeNumericalProperties:
    """Tests for numerical properties and precision."""

    def test_numerical_precision(self, grid_config, device_setup):
        """Verify numerical precision is maintained during exchange."""
        H = grid_config['H']
        W = grid_config['W']
        halo = grid_config['halo']
        ndev = device_setup['ndev']
        halo_copy_pm = device_setup['halo_copy_pm']

        # Use values with decimal precision
        grids = []
        for dev_id in range(ndev):
            grid = jnp.zeros((H, W + 2*halo), dtype=jnp.float32)
            # Use precise decimal value
            value = float(dev_id) + 0.123456789
            grid = grid.at[:, halo:halo+W].set(value)
            grids.append(grid)
        grids = jnp.stack(grids, axis=0)

        roles = jnp.arange(ndev, dtype=jnp.int32)
        result = halo_copy_pm(grids, roles)
        result_cpu = jax.device_get(result)

        L = halo
        R = halo + W

        # Check precision maintained in exchange
        if ndev > 1:
            left_halo = result_cpu[1, :, 0:L]
            expected = 0.123456789  # Device 0's value
            # Use relative tolerance for float32
            assert np.allclose(left_halo, expected, rtol=1e-6), \
                f"Numerical precision not maintained. Expected {expected}, got {left_halo[0, 0]}"

    def test_zero_values_preserved(self, grid_config, device_setup):
        """Verify zero values are correctly handled and preserved."""
        H = grid_config['H']
        W = grid_config['W']
        halo = grid_config['halo']
        ndev = device_setup['ndev']
        halo_copy_pm = device_setup['halo_copy_pm']

        # All zeros in interior
        grids = jnp.zeros((ndev, H, W + 2*halo), dtype=jnp.float32)

        roles = jnp.arange(ndev, dtype=jnp.int32)
        result = halo_copy_pm(grids, roles)
        result_cpu = jax.device_get(result)

        # All values should remain zero
        assert np.allclose(result_cpu, 0.0, atol=TOLERANCE), \
            "Zero values were not preserved correctly"


# ============================================================================
# Integration Test
# ============================================================================

def test_full_halo_exchange_integration(grid_config, device_setup, constant_test_grids):
    """
    Comprehensive integration test verifying all aspects of halo exchange.

    This test combines multiple checks:
    - Shape and dtype preservation
    - Interior preservation
    - Correct neighbor data exchange
    - Proper boundary handling
    """
    H = grid_config['H']
    W = grid_config['W']
    halo = grid_config['halo']
    ndev = device_setup['ndev']
    halo_copy_pm = device_setup['halo_copy_pm']

    L = halo
    R = halo + W
    roles = jnp.arange(ndev, dtype=jnp.int32)

    # Perform halo exchange
    result = halo_copy_pm(constant_test_grids, roles)
    result_cpu = jax.device_get(result)

    # 1. Shape and dtype
    assert result.shape == constant_test_grids.shape
    assert result.dtype == constant_test_grids.dtype

    # 2. Verify each device comprehensively
    for dev_id in range(ndev):
        grid = result_cpu[dev_id]

        # Interior unchanged
        interior = grid[:, L:R]
        expected_interior = float(dev_id + 1)
        assert np.allclose(interior, expected_interior, atol=TOLERANCE), \
            f"Device {dev_id}: Interior values incorrect"

        # Left halo
        left_halo = grid[:, 0:L]
        if dev_id > 0:
            expected_left = float(dev_id)  # Left neighbor's interior
            assert np.allclose(left_halo, expected_left, atol=TOLERANCE), \
                f"Device {dev_id}: Left halo should contain neighbor data"
        else:
            expected_boundary = -1.0  # Original halo value
            assert np.allclose(left_halo, expected_boundary, atol=TOLERANCE), \
                f"Device {dev_id}: Left boundary should remain unchanged"

        # Right halo
        right_halo = grid[:, R:R+halo]
        if dev_id < ndev - 1:
            expected_right = float(dev_id + 2)  # Right neighbor's interior
            assert np.allclose(right_halo, expected_right, atol=TOLERANCE), \
                f"Device {dev_id}: Right halo should contain neighbor data"
        else:
            expected_boundary = -float(ndev)  # Original halo value
            assert np.allclose(right_halo, expected_boundary, atol=TOLERANCE), \
                f"Device {dev_id}: Right boundary should remain unchanged"
