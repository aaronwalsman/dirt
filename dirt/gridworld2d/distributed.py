from typing import Tuple, Dict, List

import jax.numpy as jnp
from jax import lax
from mechagogue.static import static_functions

AXIS = "mesh"


def _make_perms(tile_rows: int, tile_cols: int) -> Dict[str, List[Tuple[int, int]]]:
    def idx(r: int, c: int) -> int:
        return r * tile_cols + c

    perms = {
        "right": [],
        "left": [],
        "down": [],
        "up": [],
        "down_right": [],
        "down_left": [],
        "up_right": [],
        "up_left": [],
    }
    for r in range(tile_rows):
        for c in range(tile_cols):
            src = idx(r, c)
            perms["right"].append((src, idx(r, (c + 1) % tile_cols)))
            perms["left"].append((src, idx(r, (c - 1) % tile_cols)))
            perms["down"].append((src, idx((r + 1) % tile_rows, c)))
            perms["up"].append((src, idx((r - 1) % tile_rows, c)))
            perms["down_right"].append((src, idx((r + 1) % tile_rows, (c + 1) % tile_cols)))
            perms["down_left"].append((src, idx((r + 1) % tile_rows, (c - 1) % tile_cols)))
            perms["up_right"].append((src, idx((r - 1) % tile_rows, (c + 1) % tile_cols)))
            perms["up_left"].append((src, idx((r - 1) % tile_rows, (c - 1) % tile_cols)))
    return perms


def make_distributed_halo_grid(
    halo: int,
    tile_dimensions: Tuple[int, int],
    fill_value=0,
):
    """
    Create a 2D halo exchange helper for pmap-tiled grids.

    Assumes the input grid already includes halo cells on all sides:
    shape ... x (H + 2*halo) x (W + 2*halo).
    """
    tile_rows, tile_cols = tile_dimensions
    perms = _make_perms(tile_rows, tile_cols)

    def _fill_like(x):
        return jnp.full_like(x, fill_value)

    @static_functions
    class DistributedGrid:
        def halo_padded_grid(distributed_grid):
            if halo == 0:
                return distributed_grid

            dev = lax.axis_index(AXIS)
            dev_row = dev // tile_cols
            dev_col = dev % tile_cols

            has_up = dev_row > 0
            has_down = dev_row < (tile_rows - 1)
            has_left = dev_col > 0
            has_right = dev_col < (tile_cols - 1)

            total_h = distributed_grid.shape[-2]
            total_w = distributed_grid.shape[-1]

            top = halo
            bottom = total_h - halo
            left = halo
            right = total_w - halo

            # Edge strips from the interior
            top_real = distributed_grid[..., top : top + halo, left:right]
            bottom_real = distributed_grid[..., bottom - halo : bottom, left:right]
            left_real = distributed_grid[..., top:bottom, left : left + halo]
            right_real = distributed_grid[..., top:bottom, right - halo : right]

            # Corner blocks from the interior
            top_left_real = distributed_grid[..., top : top + halo, left : left + halo]
            top_right_real = distributed_grid[..., top : top + halo, right - halo : right]
            bottom_left_real = distributed_grid[..., bottom - halo : bottom, left : left + halo]
            bottom_right_real = distributed_grid[..., bottom - halo : bottom, right - halo : right]

            # Vertical exchange (fill top/bottom ghost rows)
            payload_down = lax.cond(
                has_down, lambda _: bottom_real, lambda _: _fill_like(bottom_real), operand=None
            )
            from_up = lax.ppermute(payload_down, axis_name=AXIS, perm=perms["down"])

            payload_up = lax.cond(
                has_up, lambda _: top_real, lambda _: _fill_like(top_real), operand=None
            )
            from_down = lax.ppermute(payload_up, axis_name=AXIS, perm=perms["up"])

            distributed_grid = distributed_grid.at[
                ..., 0:top, left:right
            ].set(from_up)
            distributed_grid = distributed_grid.at[
                ..., bottom : bottom + halo, left:right
            ].set(from_down)

            # Horizontal exchange (fill left/right ghost columns)
            payload_right = lax.cond(
                has_right, lambda _: right_real, lambda _: _fill_like(right_real), operand=None
            )
            from_left = lax.ppermute(payload_right, axis_name=AXIS, perm=perms["right"])

            payload_left = lax.cond(
                has_left, lambda _: left_real, lambda _: _fill_like(left_real), operand=None
            )
            from_right = lax.ppermute(payload_left, axis_name=AXIS, perm=perms["left"])

            distributed_grid = distributed_grid.at[
                ..., top:bottom, 0:left
            ].set(from_left)
            distributed_grid = distributed_grid.at[
                ..., top:bottom, right : right + halo
            ].set(from_right)

            # Diagonal exchange (fill ghost corners)
            has_down_right = has_down & has_right
            has_down_left = has_down & has_left
            has_up_right = has_up & has_right
            has_up_left = has_up & has_left

            payload_down_right = lax.cond(
                has_down_right,
                lambda _: bottom_right_real,
                lambda _: _fill_like(bottom_right_real),
                operand=None,
            )
            from_up_left = lax.ppermute(
                payload_down_right, axis_name=AXIS, perm=perms["down_right"]
            )

            payload_down_left = lax.cond(
                has_down_left,
                lambda _: bottom_left_real,
                lambda _: _fill_like(bottom_left_real),
                operand=None,
            )
            from_up_right = lax.ppermute(
                payload_down_left, axis_name=AXIS, perm=perms["down_left"]
            )

            payload_up_right = lax.cond(
                has_up_right,
                lambda _: top_right_real,
                lambda _: _fill_like(top_right_real),
                operand=None,
            )
            from_down_left = lax.ppermute(
                payload_up_right, axis_name=AXIS, perm=perms["up_right"]
            )

            payload_up_left = lax.cond(
                has_up_left,
                lambda _: top_left_real,
                lambda _: _fill_like(top_left_real),
                operand=None,
            )
            from_down_right = lax.ppermute(
                payload_up_left, axis_name=AXIS, perm=perms["up_left"]
            )

            distributed_grid = distributed_grid.at[..., 0:top, 0:left].set(from_up_left)
            distributed_grid = distributed_grid.at[
                ..., 0:top, right : right + halo
            ].set(from_up_right)
            distributed_grid = distributed_grid.at[
                ..., bottom : bottom + halo, 0:left
            ].set(from_down_left)
            distributed_grid = distributed_grid.at[
                ..., bottom : bottom + halo, right : right + halo
            ].set(from_down_right)

            return distributed_grid

    return DistributedGrid
