# QCPINN Experiments

This repository is an extended experimental version of **QCPINN: Quantum-Classical Physics-Informed Neural Networks** for solving PDEs.

It is based on the original repository [afrah/QCPINN.git](https://github.com/afrah/QCPINN.git), which accompanies the paper [QCPINN: Quantum-Classical Physics-Informed Neural Networks for Solving PDEs](https://iopscience.iop.org/article/10.1088/2632-2153/ae1c91). This fork keeps the core PINN and quantum-classical PINN components, and adds exact-solution experiments, repeated-run evaluation, RAR comparisons, merged curve plotting, and final metric summaries.

## Project Structure

```text
QCPINN/
|-- checkpoints/        # Training outputs, merged curves, plots, and summary CSVs
|-- scripts/            # Metric extraction, normalization, merging, and summaries
|-- src/
|   |-- data/           # Exact datasets and PDE data generators
|   |-- nn/             # Classical and quantum neural-network solvers
|   |-- trainer/        # Training entry points for exact PDE experiments
|   |-- utils/          # Plotting, metrics, logging, and helper utilities
|-- qcpinn.yaml         # Conda environment file
|-- final_metrics_summary.csv
`-- README.md
```

## Installation

```bash
git clone https://github.com/lile0329/QPINN-RAR.git
cd QCPINN-RAR
conda env create -f qcpinn.yaml
conda activate qcpinn
```

## Experiments

The active workflow focuses on three exact PDE problems:

- `burgers_exact`
- `diffusion_exact`
- `heat3d_exact`

The comparison uses four method labels:

| Method | Solver setting | RAR setting |
|---|---|---|
| `PINN` | `solver = "Classical2"` | `use_rar = False` |
| `PINN-RAR` | `solver = "Classical2"` | `use_rar = True` |
| `QPINN` | `solver = "DV"` | `use_rar = False` |
| `QPINN-RAR` | `solver = "DV"` | `use_rar = True` |

These labels are parsed from each run's `output.log` by the postprocessing scripts. Training outputs are written under:

```text
checkpoints/<equation>/<run_id>/
```

## Training

Classical PINN entry points:

```bash
python -m src.trainer.burgers_exact_hybrid_trainer
python -m src.trainer.diffusion_exact_hybrid_trainer
python -m src.trainer.heat3d_exact_hybrid_trainer
```

Set `solver = "Classical2"` in the selected trainer to run the PINN family. Set `use_rar = True` to run the corresponding RAR version.

Hybrid quantum-classical QPINN entry points:

```bash
python -m src.trainer.burgers_exact_hybrid_trainer
python -m src.trainer.diffusion_exact_hybrid_trainer
python -m src.trainer.heat3d_exact_hybrid_trainer
```

Set `solver = "DV"` in the selected trainer to run the QPINN family. Set `use_rar = True` to run `QPINN-RAR`.

The older exact training modules are still available:

```bash
python -m src.trainer.burgers_exact_train
python -m src.trainer.diffusion_exact_train
python -m src.trainer.heat3d_exact_train
```

## Postprocessing

Run these scripts after training to rebuild per-run curves, normalize iterations, merge repeated runs, and summarize the final metrics:

```bash
python scripts/extract_checkpoints_metrics.py --overwrite
python scripts/normalize_training_curve_iterations.py --checkpoints_dir checkpoints
python scripts/merge_10runs_training_curves.py --checkpoints_dir checkpoints --out_dir checkpoints
python scripts/summarize_final_metrics_by_method.py
```

`scripts/summarize_final_metrics_by_method.py` writes `final_metrics_summary.csv` by default. It reports the mean and standard deviation for each equation and method using the final valid row from each run's `training_curve_iteration_normalized.csv`.

Merged input CSV files are saved under each equation folder, for example:

```text
checkpoints/burgers_exact/merged_loss.csv
checkpoints/burgers_exact/merged_rel_l2_error.csv
checkpoints/diffusion_exact/merged_loss.csv
checkpoints/diffusion_exact/merged_rel_l2_error.csv
checkpoints/heat3d_exact/merged_loss.csv
checkpoints/heat3d_exact/merged_rel_l2_error.csv
```

## Plotting

Plot four-method loss and relative-error ranges from the merged CSV files:

```bash
python checkpoints/plot_ranges_all_equations.py
```

Plot the same four methods with per-run normalized iteration, so each run ends at `iteration_normalized = 1`:

```bash
python checkpoints/plot_ranges_all_equations_normalized_iter.py
```

Plot 10 individual runs per method, plus overlay plots and mean-with-band plots:

```bash
python checkpoints/plot_10runs_training_curves_by_method_from_normiter_csv.py
```

Useful options:

```bash
python checkpoints/plot_10runs_training_curves_by_method_from_normiter_csv.py --equation heat3d_exact
python checkpoints/plot_10runs_training_curves_by_method_from_normiter_csv.py --layout by_method
python checkpoints/plot_10runs_training_curves_by_method_from_normiter_csv.py --out_dirname plots_10runs
```

The plot scripts generate PNG files inside each equation directory or under each equation's `plots_10runs/` directory.



## Notes On This Fork

Compared with the original [afrah/QCPINN.git](https://github.com/afrah/QCPINN.git), this repository is organized around repeated exact-solution experiments and statistical comparison of four methods: `PINN`, `PINN-RAR`, `QPINN`, and `QPINN-RAR`. Some modules from the upstream project are retained for compatibility, while the active analysis workflow is centered on the exact Burgers, diffusion, and 3D heat experiments.

## License

MIT [LICENSE](LICENSE)

## Reference

If you use the original QCPINN method or codebase, please cite:

```bibtex
@article{Farea:2025:MLST,
  author = {Farea, Afrah and Khan, Saiful and Celebi, Mustafa Serdar},
  title = {QCPINN: Quantum-Classical Physics-Informed Neural Networks for Solving PDEs},
  journal = {Machine Learning: Science and Technology},
  url = {http://iopscience.iop.org/article/10.1088/2632-2153/ae1c91},
  year = {2025}
}
```
