# R&D / AI Curve Parameter Estimation

## Final answer

Unknown variables:

```text
theta = 30 degrees = 0.523598776 radians
M     = 0.03
X     = 55
```

Desmos / LaTeX equation:

```latex
\left(t\cdot\cos(0.523598776)-e^{0.030000000\left|t\right|}\cdot\sin(0.3t)\sin(0.523598776)+55,\ 42+t\cdot\sin(0.523598776)+e^{0.030000000\left|t\right|}\cdot\sin(0.3t)\cos(0.523598776)\right)
```

Equivalent exact-angle form:

```latex
\left(t\cos\left(\frac{\pi}{6}\right)-e^{0.03|t|}\sin(0.3t)\sin\left(\frac{\pi}{6}\right)+55,\ 42+t\sin\left(\frac{\pi}{6}\right)+e^{0.03|t|}\sin(0.3t)\cos\left(\frac{\pi}{6}\right)\right)
```

## Method

The given curve is:

```text
x = t*cos(theta) - exp(M*|t|)*sin(0.3t)*sin(theta) + X
y = 42 + t*sin(theta) + exp(M*|t|)*sin(0.3t)*cos(theta)
```

For a trial value of `theta` and `X`, rotate the observed points back into the curve's local coordinate system:

```text
t_est = (x - X)*cos(theta) + (y - 42)*sin(theta)
w_est = -(x - X)*sin(theta) + (y - 42)*cos(theta)
```

Then the second local coordinate should satisfy:

```text
w_est = exp(M*|t_est|)*sin(0.3*t_est)
```

The script searches the allowed parameter ranges:

```text
0 deg < theta < 50 deg
-0.05 < M < 0.05
0 < X < 100
```

It first performs a coarse search, then refines the best candidates using coordinate descent. The final parameters are selected by minimizing the mean squared residual in the transformed coordinate system.

## Verification

Using:

```text
theta = 30 deg
M = 0.03
X = 55
```

the reconstruction error on `xy_data.csv` is:

```text
mean xy L1 error = 2.06e-05
max xy L1 error  = 5.53e-05
total xy L1      = 0.0308 over 1500 points
```

The small nonzero error is consistent with rounding in the CSV data.

## Run the code

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python fit_curve.py
```

Expected result:

```text
theta_rad: about 0.523598776
theta_deg: about 30
M: about 0.03
X: about 55
```
