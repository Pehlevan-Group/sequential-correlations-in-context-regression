# Sequential Correlations Change In-Context Learning

This respository contains all code needed to reproduce figures and experiments for our upcoming paper "Sequential Correlations Change In-Context Learning: \\ Effective Context Length and Architectural Mismatch" from _Mary Letey, Yue M. Lu, Cengiz Pehlevan, and Jacob Zavatone-Veth._ Paper to be released shortly.

## Repo organisation
This repository will be organised as follows

- `theory_base`: all code for running simulations of the theory model, i.e. computing the reduced-linear-attention parameter matrix $\Gamma^*$ from data.
- `transformer_base`: all basic architecture specs and training code for the models we train, i.e. full parameter linear attention and various softmax / mlp architectures.
- `specific_figures`: saved data from our runs that generate our figures, as well as instructions for regenerating this data from scratch using `theory_base` and `transformer_base`.

## Environment

Before running anything, you will need to set up an environment that has all the packages we use. 

