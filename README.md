# NLP benchmark for multiple GPUs

This is a simple script to quickly benchmark a multiple-GPU system on an NLP task.

Specifically, it fine-tunes an English [BERT-Large](https://huggingface.co/bert-large-cased) language model on three [GLUE](https://gluebenchmark.com/) tasks:
- QQP
- MNLI
- QNLI

## Instructions

NB: run `accelerate config` if you did not use the `accelerate` library before and have just installed it.

Simply run `ai_nlp_benchmark.sh`. It will automatically download the BERT model and the GLUE datasets.

Then it will sequentially fine-tune the model on each dataset and save the resulting models and their evaluation results to corresponding sub-directories

The code uses 4 GPUs by default, one can change it in the `ai_nlp_benchmark.sh` script.
The default per-device batch size is 32, can also be decreased if it is too large for the devices under evaluation.

Every task should take about 1-3 hours.

## Results
The evaluation results can be found in the corresponding sub-directories (for example, `qnli_results/all_results.json`).

The scores should not be significantly different from the following:

- QQP: accuracy 0.91, F1 0.88
- MNLI: accuracy 0.87
- QNLI: accuracy 0.92

