# BCHW Tensor Flattening and Reconstruction

## Objective

This assignment implements the transformations

```text
B x C x H x W  ->  B x (C*H*W)  ->  B x C x H x W
```

using explicit indexing. The C++ implementation is in [`tensor_flatten.cpp`](./tensor_flatten.cpp), and the Python implementation is in [`tensor_flatten.py`](./tensor_flatten.py). Neither algorithm uses a library reshape/flatten operation to perform the required transformations. The Python program also uses NumPy reshape only for the separately labeled reference check, after the explicit implementation.

## Index mapping

For BCHW row-major storage, the linear input offset is

```text
input_offset(b,c,h,w) = b*(C*H*W) + c*(H*W) + h*W + w
```

For the two-dimensional tensor `F` with shape `B x (C*H*W)`, the corresponding element is

```text
F[b,k] = I[b,c,h,w]
k = c*(H*W) + h*W + w
```

To reconstruct, decode each flattened column index `k`:

```text
c   = k // (H*W)
rem = k % (H*W)
h   = rem // W
w   = rem % W
I_hat[b,c,h,w] = F[b,k]
```

The batch index `b` selects the row and is unchanged by either transformation.

### Pseudocode

```text
flatten(I, B, C, H, W):
    allocate F[B][C*H*W]
    for b in 0..B-1:
        for c in 0..C-1:
            for h in 0..H-1:
                for w in 0..W-1:
                    k = c*(H*W) + h*W + w
                    F[b][k] = I[b][c][h][w]
    return F

reconstruct(F, B, C, H, W):
    allocate I_hat[B][C][H][W]
    for b in 0..B-1:
        for k in 0..C*H*W-1:
            c = k // (H*W)
            rem = k % (H*W)
            h = rem // W
            w = rem % W
            I_hat[b][c][h][w] = F[b][k]
    return I_hat
```

## Manual example

Let `B=1, C=2, H=2, W=3`. The flattened row contains `C*H*W = 12` elements. Using sample values `I[0,c,h,w] = 100*c + 10*h + w`:

| BCHW element | Flattened index `k = c*6 + h*3 + w` | Flattened value |
|---|---:|---:|
| `I[0,0,0,0]` | 0 | 0 |
| `I[0,0,1,2]` | 5 | 12 |
| `I[0,1,0,0]` | 6 | 100 |
| `I[0,1,1,2]` | 11 | 112 |

Thus the complete flattened row is:

```text
[0, 1, 2, 10, 11, 12, 100, 101, 102, 110, 111, 112]
```

## Experiments and results

Both programs were run. The C++ program was compiled with `g++ -std=c++17 -Wall -Wextra -Wpedantic -O2`. The Python program ran with NumPy. Each implementation generated its own test input; the random Python tensor values and deterministic C++ values are not intended to be identical. We compared the reported error measures for the same six tensor shapes.

**Dataset note:** No MNIST or CIFAR dataset files are loaded by either program. For the first two rows, the programs generate synthetic tensors with dimensions matching a single-channel 28x28 MNIST image and a three-channel 32x32 CIFAR image. The remaining inputs are also synthetic feature maps.

| Input | B | C | H | W | Elements | Emax | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Synthetic (MNIST shape) | 1 | 1 | 28 | 28 | 784 | 0.00000 | 0.00000 |
| Synthetic (CIFAR shape) | 1 | 3 | 32 | 32 | 3,072 | 0.00000 | 0.00000 |
| Synthetic feature map 1 | 2 | 16 | 16 | 16 | 8,192 | 0.00000 | 0.00000 |
| Synthetic feature map 2 | 2 | 64 | 16 | 16 | 32,768 | 0.00000 | 0.00000 |
| Synthetic feature map 3 | 2 | 128 | 16 | 16 | 65,536 | 0.00000 | 0.00000 |
| Synthetic feature map 4 | 2 | 500 | 16 | 16 | 256,000 | 0.00000 | 0.00000 |

The C++ and Python outputs both reported `Emax = 0` and `MAE = 0` for every case, including the manual example.

## Error calculation and discussion

For every element, the programs calculate the absolute difference `abs(I - I_hat)`. They report

```text
Emax = maximum absolute difference over all elements
MAE  = sum of absolute differences / number of elements
```

Zero errors are the expected result here. Flattening and reconstruction only copy each value to and from its corresponding index; they do not perform arithmetic on tensor values or change their data type. The explicit index formulas are inverse mappings, so every tested input element returns to its original position. This shows no reconstruction loss for these tests; it does not claim that every conceivable input or implementation is error-free.

## Reproducing the experiments

With a C++17 compiler and NumPy installed:

```powershell
g++ -std=c++17 -Wall -Wextra -Wpedantic -O2 tensor_flatten.cpp -o tensor_flatten.exe
.\tensor_flatten.exe
python tensor_flatten.py
```
