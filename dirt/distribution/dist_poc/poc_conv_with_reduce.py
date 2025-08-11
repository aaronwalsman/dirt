# !/usr/bin/env python3
# JIT + lax.reduce_window
# 4-device chain (A<->B<->C<->D), column halos (width=halo), NO wrap data.
# Each step:
#   (1) Halo COPY (neighbor real -> local ghosts)
#   (2) Box-mean update using ghosts (VALID, skip physical outer borders)

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # set before importing jax

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, pmap
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

AXIS = "mesh"  # pmap axis name

# ----------------------------- Init -----------------------------

def make_init_local(nx: int, ny: int, halo: int):
    """Allocate (nx, ny + 2*halo) on each device and fill with that device's role id (0,1,2,3)."""
    shape = (nx, ny + 2 * halo)
    def _fn(role):
        val = role.astype(jnp.float32)
        return jnp.full(shape, val, dtype=jnp.float32)
    return _fn

# ------------------------- Halo COPY (collectives) -------------------------

def make_halo_copy(nx: int, ny: int, halo: int, ndev: int):
    """
    Copy neighbor *real* border strips into local ghosts (no wrap data).
      - LEFT ghost  <= LEFT neighbor's RIGHT-real
      - RIGHT ghost <= RIGHT neighbor's LEFT-real
    Ends send zeros on the wrap link to keep ppermute bijective; writes are edge-guarded.
    """
    L = halo
    R = halo + ny

    perm_right_shift = [(i, (i + 1) % ndev) for i in range(ndev)]  # send to right neighbor
    perm_left_shift  = [(i, (i - 1) % ndev) for i in range(ndev)]  # send to left  neighbor

    def halo_copy(u: jnp.ndarray, role: jnp.ndarray) -> jnp.ndarray:
        dev = lax.axis_index(AXIS)

        # Real border strips (columns)
        my_left_real  = u[:, L : L + halo]    # (nx, halo)
        my_right_real = u[:, R - halo : R]    # (nx, halo)

        # For LEFT ghost: receive LEFT neighbor's right-real via right-shift
        payload_right = lax.cond(
            dev == (ndev - 1),
            lambda _: jnp.zeros_like(my_right_real),  # zero on wrap
            lambda _: my_right_real,
            operand=None,
        )
        left_neighbor_right = lax.ppermute(payload_right, axis_name=AXIS, perm=perm_right_shift)

        # For RIGHT ghost: receive RIGHT neighbor's left-real via left-shift
        payload_left = lax.cond(
            dev == 0,
            lambda _: jnp.zeros_like(my_left_real),   # zero on wrap
            lambda _: my_left_real,
            operand=None,
        )
        right_neighbor_left = lax.ppermute(payload_left, axis_name=AXIS, perm=perm_left_shift)

        # Write ghosts (edge-guarded)
        u = lax.cond(
            role > 0,
            lambda x: x.at[:, 0:L].set(left_neighbor_right),  # left ghost
            lambda x: x,
            operand=u,
        )
        u = lax.cond(
            role < (ndev - 1),
            lambda x: x.at[:, R:R + halo].set(right_neighbor_left),  # right ghost
            lambda x: x,
            operand=u,
        )
        return u

    return halo_copy

# ------------------------- Box-mean compute (uses ghosts) ------------------

def box_mean_valid(x: jnp.ndarray, radius: int) -> jnp.ndarray:
    """
    (2r+1)x(2r+1) box mean with VALID semantics (no padding).
    Input:  (H, W)
    Output: (H-2r, W-2r)
    """
    r = int(radius)
    window_dims    = (2 * r + 1, 2 * r + 1)
    window_strides = (1, 1)
    padding        = ((0, 0), (0, 0))  # VALID
    init           = jnp.array(0, dtype=x.dtype)
    total = lax.reduce_window(x, init, lax.add, window_dims, window_strides, padding)
    return total / float(window_dims[0] * window_dims[1])

def make_update_with_ghosts_reducewindow(nx: int, ny: int, halo: int, ndev: int):
    """
    Use ghosts to update interior via (2h+1)x(2h+1) box mean (VALID).
    Rows: update only r..nx-r-1 (skip physical top/bottom).
    Cols:
      - role==0       : skip first  r interior cols (global left edge)
      - role==ndev-1  : skip last   r interior cols (global right edge)
      - interior roles: update full interior L..R-1
    Implemented by writing the full interior then restoring the skipped edge bands.
    """
    r = halo
    L, R = halo, halo + ny  # interior columns

    @jax.jit
    def update(u: jnp.ndarray, role: jnp.ndarray) -> jnp.ndarray:
        # VALID box mean over the local array (with ghosts)
        avg = box_mean_valid(u, r)  # shape: (nx-2r, ny)

        v = u
        # Map VALID result to interior rows/cols:
        # rows r..nx-r-1, cols L..R-1  ← avg[:, (L-r):(R-r)]
        v = v.at[r:nx - r, L:R].set(avg[:, (L - r):(R - r)])

        # Skip outer global columns by restoring from the old state:
        v = lax.cond(
            role == 0,
            lambda x: x.at[r:nx - r, L:L + r].set(u[r:nx - r, L:L + r]),
            lambda x: x,
            operand=v,
        )
        v = lax.cond(
            role == (ndev - 1),
            lambda x: x.at[r:nx - r, R - r:R].set(u[r:nx - r, R - r:R]),
            lambda x: x,
            operand=v,
        )
        return v

    return update

# ------------------------------ One step ------------------------------

def make_step(nx: int, ny: int, halo: int, ndev: int):
    halo_copy = make_halo_copy(nx, ny, halo, ndev)                       # comm
    update_fn = make_update_with_ghosts_reducewindow(nx, ny, halo, ndev)  # JIT compute

    def step(u, role):
        u = halo_copy(u, role)  # 1) copy halos from neighbors (collective, barrier)
        u = update_fn(u, role)  # 2) box-mean update using ghosts (skip physical edges)
        return u

    return step

# ------------------------------ CPU reference (optional) -------------------

def cpu_reference(full: np.ndarray, halo: int, steps: int) -> np.ndarray:
    """
    CPU reference on the full domain (no subdomains).
    VALID (no pad): update rows [r:H-r), cols [r:W-r).
    """
    H, W = full.shape
    r = halo
    x = full.astype(np.float32).copy()
    k = 2 * r + 1

    # helper: 2D integral image for fast box sums
    def box_mean_valid_cpu(arr: np.ndarray) -> np.ndarray:
        S = np.cumsum(np.cumsum(arr, axis=0), axis=1)
        def block_sum(ii, jj):
            A = S[ii + k - 1, jj + k - 1]
            B = S[ii - 1,     jj + k - 1] if ii > 0 else 0.0
            C = S[ii + k - 1, jj - 1]     if jj > 0 else 0.0
            D = S[ii - 1,     jj - 1]     if (ii > 0 and jj > 0) else 0.0
            return A - B - C + D
        Hv, Wv = H - 2 * r, W - 2 * r
        out = np.empty((Hv, Wv), dtype=np.float32)
        for i in range(Hv):
            for j in range(Wv):
                out[i, j] = block_sum(i, j) / float(k * k)
        return out

    for _ in range(steps):
        avg = box_mean_valid_cpu(x)                 # (H-2r, W-2r)
        x_next = x.copy()
        x_next[r:H - r, r:W - r] = avg             # write VALID region
        x = x_next
    return x

# --------------------------------- Main ---------------------------------

def show_matrix(M, title=None, vmin=None, vmax=None, cmap="viridis", center=None):
    M = np.asarray(M)
    plt.figure(figsize=(6, 4))

    # Optional: center the colormap (useful if data has positive & negative values)
    norm = None
    if center is not None:
        vmin_ = M.min() if vmin is None else vmin
        vmax_ = M.max() if vmax is None else vmax
        norm = TwoSlopeNorm(vmin=vmin_, vcenter=center, vmax=vmax_)

    im = plt.imshow(
        M,
        cmap=cmap,          # <- pick your palette here
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        norm=norm,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im)    

    if title:
        plt.title(title)
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)
    else:
        safe = "matrix"

    plt.xlabel("column")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(f"{safe}.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32, help="Rows per device (height)")
    ap.add_argument("--ny", type=int, default=32, help="Cols per device interior (width)")
    ap.add_argument("--halo", type=int, default=3, help="Neighborhood radius (also halo width)")
    ap.add_argument("--steps", type=int, default=2, help="Time steps")
    ap.add_argument("--ndev", type=int, default=4, help="Devices in chain")
    ap.add_argument("--debug", action="store_true", help="Print small arrays / extra checks")
    ap.add_argument("--compare", action="store_true", help="Run CPU reference and compare")
    args = ap.parse_args()

    assert jax.device_count() >= args.ndev, f"Need >= {args.ndev} devices; found {jax.device_count()}"
    devices = jax.devices()[:args.ndev]
    print("Using devices:", devices)

    roles = jnp.arange(args.ndev, dtype=jnp.int32)

    # Initialize per-device arrays with role id
    init_pm = pmap(make_init_local(args.nx, args.ny, args.halo),
                   in_axes=0, axis_name=AXIS, devices=devices)
    U = init_pm(roles)

    # Compile step
    step_pm = pmap(make_step(args.nx, args.ny, args.halo, args.ndev),
                   in_axes=(0, 0), axis_name=AXIS, devices=devices)

    # Run steps
    for _ in range(args.steps):
        U = step_pm(U, roles)

    # Gather interiors and build global field
    shards = [jax.device_get(U[i]) for i in range(args.ndev)]
    L, R = args.halo, args.halo + args.ny
    gpu_global = np.concatenate([np.array(s[:, L:R]) for s in shards], axis=1)

    print(f"GPU result shape (global interior): {gpu_global.shape}")

    if args.compare:
        full_init = np.concatenate(
            [np.full((args.nx, args.ny), float(i), dtype=np.float32) for i in range(args.ndev)],
            axis=1
        )
        print("CPU Initial full array shape:", full_init.shape)
        cpu_final = cpu_reference(full_init, args.halo, args.steps)
        diff = gpu_global - cpu_final
        l2 = float(np.linalg.norm(diff))
        linf = float(np.max(np.abs(diff)))
        print(f"CPU vs GPU after {args.steps} step(s): L2={l2:.6e}, Linf={linf:.6e}")

    if args.debug and args.nx <= 20 and args.ny <= 20:
        print("\nConcatenated GPU interior:\n", gpu_global)

    show_matrix(gpu_global, title="GPU", cmap="Spectral")
    if args.compare:
        show_matrix(cpu_final, title="CPU_ref", cmap="Spectral")
        show_matrix(diff, title="GPU_CPU_diff", cmap="seismic", center=0.0)

if __name__ == "__main__":
    main()

# python poc_conv_with_reduce.py --nx 100 --ny 100 --halo 5 --steps 1000 --compare