# Is synthetic data private by default?

Code, results and the `sdaudit` toolkit for the paper

> **Is synthetic data private by default? An empirical audit of twelve claims about synthetic tabular data**
> Mohammad Haris Mushtaq and Naveed Yousuf, National University of Technology (NUTECH), Islamabad, Pakistan.
> Manuscript submitted to the *Journal of Information Security and Applications*. Citation details and a DOI will be added here after acceptance.

## What this paper is about

Companies are told that synthetic data lets them use personal data for AI without privacy risk. We collected twelve claims of this kind and tested them on public tables of up to 5,000 rows. We compared ten release methods on eight datasets in more than 10,000 generator runs. The paper reports which claims held, which did not, and why average privacy tests can miss leakage on unusual people.

In short: synthetic data is not private by default, and each release needs its own audit.

## What is in this repository

| Folder or file | What it holds |
|---|---|
| `sdaudit.py` | The audit toolkit (version 0.3.0). Main entry point: `audit(train, holdout, test, release, meta)`. |
| `notebooks/phase1/` | Phase I notebook (pre-specified tests), executed, with all outputs. |
| `notebooks/phase2/` | Phase II notebooks 00 to 06, executed, with all outputs. Notebook 06 holds the stress tests S1 to S4. `outputs2/` has every result table (CSV), figures and example synthetic releases. |
| `stress_test_s2b/` | The add/remove canary audit of SmartNoise MST (stress test S2, second version). |
| `revision/` | Analyses added after a simulated peer review: null controls, dataset-level and mechanism-level tests, MST with a public schema, the like-for-like pooled test, the planted-leakage test, and the scripts that build the paper tables. `MANIFEST.sha256` lists a SHA-256 hash for every file in this repository. |

The ten release methods are Identity (control), Independent, Gaussian copula, CART, CTGAN, TVAE, a simplified diffusion model, a small autoregressive transformer, MST (SmartNoise-synth) and Mondrian k-anonymity. The toolkit also contains the attacks, the Anonymeter and distance-to-closest-record checks, the fairness checks and the pass/fail gates described in the paper.

## Data

All eight datasets are public. The code downloads them on first use into a `data_raw/` folder and prints a SHA-256 hash for each file. The datasets are not stored in this repository.

- Adult, German Credit and Pima: UCI Machine Learning Repository.
- Credit-card fraud: Kaggle (Dal Pozzolo et al., 2015).
- COMPAS, Taiwan credit default, Bank Marketing and Diabetes-130: the cleaned collection of Le Quy et al. (2022).

## Setup

Python 3.11 or newer is needed. A GPU is not needed; all experiments ran on a 2-vCPU machine.

```bash
git clone https://github.com/Harismushtaq12/synthetic-data-audit.git
cd synthetic-data-audit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `import sdaudit` work from any folder
```

## Quick start

```python
import sdaudit as sa

df, meta = sa.load("adult")        # downloads and checks the file
# train, holdout, test = ...        # see notebooks/phase2/00_toolkit_selftests_hypotheses_executed.ipynb
# report = sa.audit(train, holdout, test, release, meta)
```

Notebook 00 shows the toolkit end to end, including its positive and negative control tests.

## Reproducing the results

- **Phase I and II notebooks.** Open them in Jupyter. They are saved with their outputs, so you can read them without running anything. Re-running takes about 2.5 hours (Phase I) and 11.6 hours (Phase II) on 2 CPU cores.
- **Revision analyses.** Run the scripts from inside the `revision/` folder. Examples:

  ```bash
  cd revision
  python scripts/null_controls.py adult,german,fraud,pima,compas,credit,bank,diabetes all
  python scripts/mst_public.py main compas,credit,bank,diabetes a2    # MST with a public schema
  python scripts/planted_run.py adult,bank a 2                        # planted-leakage test
  python scripts/pooled_run.py                                        # like-for-like pooled test
  python scripts/r8_dataset_level.py && python scripts/r9_mechanism_level.py
  python scripts/build_tables.py && python scripts/build_tables2.py   # LaTeX tables into tables_tex/
  ```
- **Stress test S2b.** `stress_test_s2b/s2b_addremove_audit.py` needs `smartnoise-synth==1.0.8` and takes about one hour on 2 CPU cores. `make_revised_figures.py` redraws the revised figures from the saved CSV tables.
- **SmartNoise behaviour.** `revision/scripts/smartnoise_domain_demo.py` is a short script that shows, for SmartNoise-synth 1.0.8, that the category lists come from the private data unless a public schema is given.

Seeds control the splits, every generator, the downstream models and the attacks.

## Software versions

The Phase I and Phase II notebooks ran with Python 3.11, scikit-learn 1.8, SDV 1.38, PyTorch 2.14, SmartNoise-synth 1.0.8, Anonymeter 1.1.0 and CodeCarbon 3.3.1 (toolkit versions 0.2.0 and 0.2.1). The revision scripts ran with Python 3.13, SDV 2.2.0 and toolkit version 0.3.0. Version 0.3.0 adds the tie-aware TPR@1% and the duplicate ceiling, makes the outcome-gap gate opt-in, and adapts the SDV calls to SDV 2.x. The Phase I and Phase II results were not rerun with it. `requirements.txt` lists the versions of the revision runs.

## Notes

- The executed notebooks are kept exactly as they were run. Their first cells still carry the title and author list of the earlier, unpublished draft whose claims we test.
- Some results come from one to four seeds and are exploratory. The paper states which ones.
- The paper and this code were prepared with help from Claude (Anthropic). Every number in the paper was checked against the saved outputs. The paper's declaration gives the details.

## Licence

The code is released under the MIT licence (see `LICENSE`). The datasets keep their own licences.

## Contact

Mohammad Haris Mushtaq (corresponding author): mohammadharismushtaq@gmail.com
