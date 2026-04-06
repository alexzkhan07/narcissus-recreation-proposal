# Source Notes

## Primary Sources

- Official repository: [reds-lab/Narcissus](https://github.com/reds-lab/Narcissus)
- Paper: [arXiv:2204.05255](https://arxiv.org/abs/2204.05255)

## Working Summary

The paper title is *Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information*.

From the paper abstract and repository README, the key idea is a clean-label backdoor attack that works with limited information about the training data.

The upstream README highlights a quick-start workflow using ResNet-18 with CIFAR-10 as the target dataset and Tiny ImageNet as the auxiliary dataset in its default notebook example.

## Reproduction Focus

The most defensible first milestone is to reproduce one core experiment from the official implementation before attempting:

- broader hyperparameter sweeps
- defense evaluation
- physical-world trigger testing

## Open Setup Questions

- Which dataset/model path in the repository is the fastest complete reproduction target?
- Are there pretrained checkpoints or preprocessing steps that materially affect the result?
- What hardware assumptions are baked into the original scripts?
- Which metrics in the paper are most realistic for a first-pass recreation?
