### Data collection for single-step

Sample setup for all data.

```console
python script/tabletop-vanilla/vanilla_collect_initial_state.py -cf data/tabletop_vanilla_answer_tp/collect/init/init.yaml
python script/tabletop-vanilla/vanilla_collect_initial_state.py -cf data/tabletop_vanilla_answer/collect/init/init.yaml
```

Collect human clarification and prompts for genertaing multiple choices. 

```console
python script/tabletop-vanilla/vanilla_collect_mc_pre.py -cf data/tabletop_vanilla_answer/collect/mc_pre/mc_pre.yaml
```

Prompt LM for generating multiple choices.

```console
python data/query_lm.py -cf data/tabletop_vanilla_answer/collect/mc_pre/query.yaml
```

Collect prompts for answering multiple choices, and collect human label.

```console
python script/tabletop-vanilla/vanilla_collect_mc_post.py -cf data/tabletop_vanilla_answer_tp/collect/mc_post/mc_post.yaml
python script/tabletop-vanilla/vanilla_collect_mc_post.py -cf data/tabletop_vanilla_answer_tp/collect/mc_post/mc_post.yaml
```

Prompt LM for answering multiple choices.

```console
python data/query_lm.py -cf data/tabletop_vanilla_answer_tp/collect/mc_post/query.yaml
python data/query_lm.py -cf data/tabletop_vanilla_answer/collect/mc_post/query.yaml
```

Process logprob data from LM, check if action succeeds.

```console
python script/tabletop-vanilla/vanilla_collect_answer.py -cf data/tabletop_vanilla_answer_tp/collect/answer/answer.yaml
python script/tabletop-vanilla/vanilla_collect_answer.py -cf data/tabletop_vanilla_answer/collect/answer/answer.yaml
```

### Calibration for single step

```console
python agent/predict/conformal_predictor.py -cf data/tabletop_vanilla_answer_tp/calibrate/cfg.yaml
python agent/predict/conformal_predictor.py -cf data/tabletop_vanilla_answer/calibrate/cfg.yaml
```
