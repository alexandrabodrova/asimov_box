### Data collection for multi step

Sample setup for all data.

```console
python script/tabletop-stack/stack_collect_initial_state.py -cf data/tabletop_stack/collect/init/init.yaml
```
Collect prompts for probing uncertainty. 

```console
python script/tabletop-stack/stack_collect_probe.py -cf data/tabletop_stack/collect/probe/probe.yaml
```

Prompt LM for probing uncertainty (clarification question).

```console
python data/query_lm.py -cf data/tabletop_stack/collect/probe/query.yaml
```

Collect human clarification and prompts for genertaing multiple choices. 

```console
python script/tabletop-stack/stack_collect_mc_pre.py -cf data/tabletop_stack/collect/mc_pre/mc_pre.yaml
```

Prompt LM for generating multiple choices.

```console
python data/query_lm.py -cf data/tabletop_stack/collect/mc_pre/query.yaml
```

Collect prompts for answering multiple choices, and collect human label.

```console
python script/tabletop-stack/stack_collect_mc_post.py -cf data/tabletop_stack/collect/mc_post/mc_post.yaml
```

Prompt LM for answering multiple choices.

```console
python data/query_lm.py -cf data/tabletop_stack/collect/mc_post/query.yaml
```

Process logprob data from LM, check if action succeeds (only continue next step if succeeds).

```console
python script/tabletop-stack/stack_collect_answer.py -cf data/tabletop_stack/collect/answer/answer.yaml
```

# Calibration for multi step
