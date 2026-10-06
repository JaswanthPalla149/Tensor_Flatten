"""
Tensor Flattening and Reconstruction (BCHW <-> B x (CHW))
Assignment Implementation from Scratch.
"""

import numpy as np


def flatten_bchw_to_2d(tensor: np.ndarray) -> np.ndarray:
    """Converts a 4D tensor I of shape (B, C, H, W) to a 2D tensor F of shape (B, C*H*W)

    using explicit index arithmetic.
    """
    B, C, H, W = tensor.shape
    CHW = C * H * W
    HW = H * W

    # Allocate 2D output matrix (B, CHW)
    flattened = np.empty((B, CHW), dtype=tensor.dtype)

    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    # Explicit 1D linearized column index within row b
                    k = c * HW + h * W + w
                    flattened[b, k] = tensor[b, c, h, w]

    return flattened


def reconstruct_2d_to_bchw(
    flattened: np.ndarray, shape: tuple[int, int, int, int]
) -> np.ndarray:
    """Reconstructs a 4D tensor I_hat of shape (B, C, H, W) from a 2D tensor F of shape (B, C*H*W)

    using quotient and modulo index decoding.
    """
    B, C, H, W = shape
    CHW = C * H * W
    HW = H * W

    assert (
        flattened.shape[0] == B and flattened.shape[1] == CHW
    ), "Dimension mismatch"

    # Allocate 4D output tensor (B, C, H, W)
    reconstructed = np.empty((B, C, H, W), dtype=flattened.dtype)

    for b in range(B):
        for k in range(CHW):
            # Inverse coordinate decoding from linear index k
            c = k // HW
            rem = k % HW
            h = rem // W
            w = rem % W

            reconstructed[b, c, h, w] = flattened[b, k]

    return reconstructed


def compute_errors(
    original: np.ndarray, reconstructed: np.ndarray
) -> tuple[float, float]:
    """Computes Maximum Absolute Error (E_max) and Mean Absolute Error (MAE)."""
    diff = np.abs(original - reconstructed)
    emax = float(np.max(diff))
    mae = float(np.mean(diff))
    return emax, mae


def run_experiments():
    print("=" * 80)
    print("MANUAL INDEXING EXAMPLE (B=1, C=2, H=2, W=3)")
    print("=" * 80)
    B, C, H, W = 1, 2, 2, 3
    sample = np.zeros((B, C, H, W), dtype=np.float64)
    for c in range(C):
        for h in range(H):
            for w in range(W):
                sample[0, c, h, w] = c * 100 + h * 10 + w

    flat_sample = flatten_bchw_to_2d(sample)
    recon_sample = reconstruct_2d_to_bchw(flat_sample, (B, C, H, W))

    print(f"  I[0,0,0,0] -> F[0,0]  = {flat_sample[0, 0]}")
    print(f"  I[0,0,1,2] -> F[0,5]  = {flat_sample[0, 5]}")
    print(f"  I[0,1,0,0] -> F[0,6]  = {flat_sample[0, 6]}")
    print(f"  I[0,1,1,2] -> F[0,11] = {flat_sample[0, 11]}")
    print(f"  Flattened row: {flat_sample[0].tolist()}")
    emax, mae = compute_errors(sample, recon_sample)
    print(f"  Manual example verification: Emax = {emax:.4f}, MAE = {mae:.4f}\n")

    # Experiments matching Table 1 in Assignment
    configs = [
        ("Synthetic (MNIST shape)", (1, 1, 28, 28)),
        ("Synthetic (CIFAR shape)", (1, 3, 32, 32)),
        ("Synthetic fmap1", (2, 16, 16, 16)),
        ("Synthetic fmap2", (2, 64, 16, 16)),
        ("Synthetic fmap3", (2, 128, 16, 16)),
        ("Synthetic fmap4", (2, 500, 16, 16)),
    ]

    print("=" * 80)
    print("EXPERIMENTAL RESULTS TABLE (Table 1)")
    print("=" * 80)
    print(
        f"{'Dataset/Input':<20} {'B':>4} {'C':>6} {'H':>6} {'W':>6} {'Emax':>18} {'MAE':>18}"
    )
    print("-" * 80)

    np.random.seed(42)
    for name, shape in configs:
        # Generate representative tensor data
        tensor = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float64)

        flat = flatten_bchw_to_2d(tensor)
        recon = reconstruct_2d_to_bchw(flat, shape)
        emax, mae = compute_errors(tensor, recon)

        # Optional reference verification using library reshape (for sanity check only)
        ref_flat = tensor.reshape(shape[0], -1)
        ref_recon = ref_flat.reshape(shape)
        assert np.array_equal(
            flat, ref_flat
        ), "Flattening does not match reference!"
        assert np.array_equal(
            recon, ref_recon
        ), "Reconstruction does not match reference!"

        print(
            f"{name:<20} {shape[0]:>4} {shape[1]:>6} {shape[2]:>6} {shape[3]:>6} {emax:>18.12f} {mae:>18.12f}"
        )
    print("-" * 80)


if __name__ == "__main__":
    run_experiments()
