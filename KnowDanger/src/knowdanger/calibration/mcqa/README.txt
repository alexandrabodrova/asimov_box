
KnowDanger Dataset Builder & Labeler
====================================

  (1) Build an MCQA dataset for KnowDanger safety calibration (KnowNo-style).
  (2) Interactively label the SAFE options per row (for CP calibration later).

Files
-----
- schemas.py
- adapters/examples_loader.py
- scorer_knowno.py
- build_dataset.py
- label_dataset.py

Build a dataset
---------------
Merge scene modules and (optionally) compute scores now:

python /mnt/data/knowdanger_build/build_dataset.py   --modules /mnt/data/example2_breakroom.py /mnt/data/example1_hazard_lab.py   --out /mnt/data/knowdanger_build/combined_raw.jsonl

With scores using the (currently dummy) KnowNo-like scorer:

python /mnt/data/knowdanger_build/build_dataset.py   --modules /mnt/data/example2_breakroom.py   --out /mnt/data/knowdanger_build/break_with_scores.jsonl   --scorer score_knowno_like:score_all

Label the dataset
-----------------
python /mnt/data/knowdanger_build/label_dataset.py   --in  /mnt/data/knowdanger_build/break_with_scores.jsonl   --out /mnt/data/knowdanger_build/break_labeled.jsonl

Resume labeling later with --resume.

Next (calibration)
------------------
Use your CP script (β-mapping + dataset-conditional quantile) to compute q_hat:

python /mnt/data/knowdanger_cp/compute_calibration.py   --calib /mnt/data/knowdanger_build/break_labeled.jsonl   --out   /mnt/data/knowdanger_build/break_calibration.json   --target-coverage 0.90   --delta 0.01

At runtime, include any option with score >= 1 - q_hat; act if singleton; ask if multi; halt if empty.
