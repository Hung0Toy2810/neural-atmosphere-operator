# Cleaning manifest

This directory is a new submission copy. The development trees were not
modified. The package excludes Git metadata, notebooks, caches, logs, cloud
monitoring scripts, temporary files, raw ERA5 stores, audit scratch work,
TensorBoard/W&B outputs, diagnostic checkpoints, `last.pt`, initialization
snapshots, `last.pt`, and boundary/initialization checkpoints.

The two retained checkpoints are the validation-best artifacts for Stage 1 and
Stage 2. No intermediate checkpoint or discarded experiment is included.
Training statistics and static forcing are retained because they are required
to reproduce inference and physical-unit metrics.

All executable defaults in the curated submission use paths relative to the
submission root or explicit CLI/environment configuration. Public anonymous
GCS access uses the literal `token="anon"`; it is not a credential.
