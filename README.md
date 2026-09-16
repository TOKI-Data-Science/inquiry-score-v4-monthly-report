# Inquiry Score v4 - Monthly Model Performance Report

Generates a HTML dashboard of the Inquiry Score v4 model performance:

AUC trajectory from the OOT-valid benchmark through each aged base month (overall, by
`product_type`, by `model_type`)

## Pipeline

1. `src/module/queries/target.sql` rebuilds the aged-pool tables in Oracle
   (`t_inq_v4_bnpl_aged_pool`, `t_inq_v4_credit_aged_pool`, `t_inq_v4_lease_aged_pool`)
   and unions them into `t_inq_v4_aged_pool` (cohorts aged >= 12 months, `base_month >= 202504`).
2. `t_inq_v4_aged_pool` is loaded into pandas.
3. `reports/inquiry_score_v4_report_<YYYYMM>.html` is written.


## Benchmarks and action rules

OOT-valid AUC benchmarks live in `OOT_AUC` in `src/module/report.py`

Latest-month AUC vs. benchmark: drop >= 0.05 -> retrain / wait one month, drop >= 0.10 -> redevelop.

## Configuration (`.env`)

Copy `.env.template` to `.env` and fill in the values.

| Variable                 | Default              | Description                                  |
| ------------------------ | -------------------- | -------------------------------------------- |
| `ORACLE_*`               |                      | Oracle connection                            |
| `RUN_TARGET_SQL`         | `true`               | Rebuild aged-pool tables before reporting    |
| `AGED_POOL_TABLE`        | `t_inq_v4_aged_pool` | Table the report is built from               |
| `REPORT_DIR`             | `reports`            | Output directory for the HTML artifact       |
| `REPORT_MONTH`           | (current month)      | `YYYYMM` label used in the file name / title |
| `REPORT_START_MONTH`     | `202504`             | First base month plotted after OOT-valid     |
| `SCORE_HIGHER_IS_BETTER` | `true`               | Score direction used for AUC                 |
| `LOG_LEVEL`              | `INFO`               |                                              |

## Run locally

```bash
pip install -r requirements.txt
python script.py                                   # full run against Oracle
python script.py --skip-sql                        # reuse existing t_inq_v4_aged_pool
python script.py --input-csv pool.csv --report-month 202609 --output reports/test.html
```

Artifact: `reports/inquiry_score_v4_report_<YYYYMM>.html`

## Run with Docker

```bash
docker build -t inquiry-score-v4-monthly-report:v0.1.0 .
docker run --rm \
  --env-file=$HOME/envs/inquiry-score-v4-monthly-report.env \
  -v $HOME/reports/inquiry-score-v4:/myapp/reports \
  inquiry-score-v4-monthly-report:v0.1.0
```

## Production deployment

Production follows the TOKI project-template pattern: git tags build a Docker image on the
production server ([`.github/workflows/build-image.yml`](.github/workflows/build-image.yml)),
and a scheduled workflow runs that image as a batch job
([`.github/workflows/schedule.yml`](.github/workflows/schedule.yml), `0 6 1 * *`, or trigger
manually via "Run workflow"). Requires a self-hosted runner labeled `self-hosted, prod, container`
and `$HOME/envs/inquiry-score-v4-monthly-report.env` on that server with the `ORACLE_*` /
`REPORT_*` variables from `.env.template`.

The scheduled workflow writes the HTML report to `$HOME/reports/inquiry-score-v4/` on the runner
(persisted across runs) **and** uploads it as a GitHub Actions artifact named
`inquiry-score-v4-report-<run id>` (kept for 90 days) — download it from the workflow run's
Summary page under **Artifacts**, no server access needed.

If GitHub Actions scheduling isn't available on the server, use a host crontab instead (this
path only writes to the host directory; it does not create a workflow artifact):

```cron
0 6 1 * * docker run --rm --env-file=$HOME/envs/inquiry-score-v4-monthly-report.env -v $HOME/reports/inquiry-score-v4:/myapp/reports inquiry-score-v4-monthly-report:v0.1.0
```

## Release

Push a tag to build the image on the production server (see `.github/workflows`).

```bash
git tag v0.1.0
git push origin v0.1.0
```
