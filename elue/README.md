## [Off-Policy Meta-Reinforcement Learning with Belief-based Task Inference]

### Usage (meta-train)
```
python -O  launch_experiment.py --config <path to the config file>
```
### Usage (meta-test)
```
python -O  launch_experiment.py --config <path to the config file> --saveddir <path to the saved networks> --num_itr <#iterations at which the networks were saved>
```

### Misc
This code is based on [PEARL's code](https://github.com/katerakelly/oyster)

