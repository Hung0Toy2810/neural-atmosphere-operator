# Stage 2 — spherical corrector

This stage adds history, analytic forcing, and the spherical residual
corrector. Initialize it from a Stage 1 best checkpoint with
`python scripts/train.py --init-checkpoint PATH --data-dir PATH --run-dir PATH`.
