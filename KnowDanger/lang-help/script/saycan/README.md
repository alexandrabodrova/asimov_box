### Data collection for single step

Sample requests and collect prompts for probing uncertainty.

```console
python script/saycan/collect_data.py -cf data/saycan_palm/collect/init/cfg.yaml
```

Prompt LM for probing uncertainty (clarification question).

```console
python data/query_lm.py -cf data/saycan_palm/collect/probe/query.yaml
```

Collect human clarification and prompts for genertaing multiple choices. 

```console
python script/saycan/saycan_collect_mc_pre.py -cf data/saycan_palm/collect/mc_pre/mc_pre.yaml
```

Prompt LM for generating multiple choices.

```console
python data/query_lm.py -cf data/saycan_palm/collect/mc_pre/query.yaml
```

Collect prompts for answering multiple choices, and collect human label.

```console
python script/saycan/saycan_collect_mc_post.py -cf data/saycan_palm/collect/mc_post/mc_post.yaml
```

Prompt LM for answering multiple choices.

```console
python data/query_lm.py -cf data/saycan_palm/collect/mc_post/query.yaml
```

Process logprob data from LM, check if action succeeds.

```console
python script/saycan/saycan_collect_answer.py -cf data/saycan_palm/collect/answer/answer.yaml
```

# Calibration for multi step

Run conformal prediction.

```console
python agent/predict/conformal_predictor.py -cf data/saycan_palm/calibrate/cfg.yaml
```

