# Recreation of Narcissus Clean-Label Backdoor Attack

## Team

- Max Boyington
- Alexander Khan

## 1. Project Choice

NARCISSUS: A Practical Clean-Label Backdoor Attack with Limited Information


NARCISSUS is a clean-label backdoor attack, meaning the attack uses maliciously crafted data injected into a dataset without changing its labels (hence "clean-label"). This paper's attack was highly efficient, creating backdoors in models using only 0.05% of the training data. By optimizing the trigger to match a target class's internal features, it evades modern defenses and remains effective across a variety of datasets and real-world situations. 


Repository: [reds-lab/Narcissus](https://github.com/reds-lab/Narcissus)  
Paper: [arXiv:2204.05255](https://arxiv.org/abs/2204.05255)

## 2. Specific Task You Propose To Recreate

For this project, we propose to reproduce the primary performance claims of the NARCISSUS attack as detailed in the Comparison of attack effectiveness section of the original paper. Specifically, we will target the results presented in Table III, which demonstrates the attack's efficacy across three distinct datasets: CIFAR-10, PubFig, and Tiny-ImageNet.

Our primary focus is to verify that an adversary can achieve a high Attack Success Rate (ASR)—exceeding 85% on Tiny-ImageNet and 97% on CIFAR-10—while maintaining a clean-label constraint and an extremely low poison ratio of 0.05%. We will utilize the ResNet-18 architecture for both the surrogate and victim models to evaluate the "all-to-one" attack scenario.

## 3. Goals

Primary Goal: Successfully reproduce the all-to-one clean-label backdoor attack on the CIFAR-10 dataset, achieving an ASR comparable to the reported 97.36% using only 25 poisoned images.

Secondary Goal: Validate the transferability and scalability of the attack by recreating the results for Tiny-ImageNet and PubFig, proving the attack works even when the dataset is large or has limited diversity (like faces).

Replicate Figure 3 (Performance vs. Target-class poison ratio) to demonstrate the trade-off between Target-class Accuracy (Tar-ACC) and ASR, verifying that NARCISSUS maintains higher target accuracy than baseline attacks like BadNets-c or LC.

Learning Objective: We hope to verify the "inward-pointing" nature of the NARCISSUS trigger—specifically, how optimizing a trigger to represent the internal features of a target class makes it more persistent and harder to remove via standard defenses than arbitrarily chosen triggers

## 4. Plan To Achieve The Goals

1. **Environment setup :** Set up a Python 3.6 environment with PyTorch 1.10.1, TorchVision 0.11.2, OpenCV 4.5.3, and CUDA 11.0 to match the original authors' setup. Configure GPU access through Google Colab or SSH into a remote FSU machine. Clone the official [Narcissus repository](https://github.com/reds-lab/Narcissus) and verify the code runs end to end on CIFAR-10 with the default settings in `Narcissus.ipynb`.

2. **Source code review :**  Go over and understand the full attack pipeline: poi-warm-up surrogate training, trigger generation via POOD-assisted feature extraction, trigger insertion into the target class, and test-time query manipulation. Keep track of the role of important hyperparameters (`l_inf_r`, `surrogate_epochs`, `warmup_round`, `gen_round`, `patch_mode`) and how they influence attacks.

3. **Dataset acquisition and preprocessing :** Obtain all three datasets used in Table 3:
   - **CIFAR-10** — available directly via `torchvision.datasets`.
   - **Tiny ImageNet** — download and restructure into the format expected by the data pipeline code (used as the POOD dataset and as a target dataset in Table 3).
   - **PubFig** — download the [PubFig83](https://www.cs.columbia.edu/CAVE/databases/pubfig/) face dataset and preprocess (resize, pad, and normalize) to match the pipeline's expected input format.

   Adapt the data loading code in `Narcissus.ipynb` and `narcissus_function.py` to support Tiny ImageNet and PubFig as target datasets, including any necessary resizing, normalization, and label changes.

4. **Reproduce Table 3 — main attack results :** Use ResNet-18 to run the full Narcissus attack on each of the three datasets with a 0.05% overall poison ratio. Keep track of the clean test accuracy and the attack success rate (ASR). Look at Table 3 in the paper and see how our results stack up against the numbers in it.

5. **Reproduce Figure 4 — Ablation study :** On CIFAR-10, alter the poison ratio (e.g., 0.01%, 0.025%, 0.05%, 0.1%, 0.5%) and record ASR at each level. Plot the resulting curve and compare to Figure 4 in the paper to make sure we are getting similar results.

6. **Analysis and write-up :** Put together all the results, make tables and graphs to compare them, and write the final report. Talk about any differences between our results and the paper, possible reasons for them (like different hardware, random seeds, or hyperparameter sensitivity), and what we learned about clean label backdoor attacks.

## 5. Required Resources

- **Datasets:**
  - CIFAR-10 (60K images, 32×32, 10 classes) — via `torchvision.datasets.CIFAR10`
  - Tiny ImageNet (100K images, 64×64, 200 classes) — used both as the POOD dataset for surrogate training and as a target dataset for Table 3
  - PubFig83 (face images, ~60 identities) — downloaded from the [Columbia CAVE lab](https://www.cs.columbia.edu/CAVE/databases/pubfig/)

- **Compute:**
  - GPU with at least 8 GB VRAM. Surrogate model training and trigger generation are the most compute intensive steps. Estimated ~2–4 GPU-hours per dataset for a full attack run.
  - Access to FSU HPC cluster via SSH or Google Colab TPU as a fallback.

- **Libraries and frameworks:**
  - Python 3.6, PyTorch 1.10.1, TorchVision 0.11.2, OpenCV 4.5.3, CUDA 11.0
  - NumPy, Matplotlib (for plotting Figure 4 and result visualizations)
  - Jupyter Notebook (for running and adapting `Narcissus.ipynb`)

- **Pretrained models / checkpoints:**
  - The attack pipeline trains its own surrogate from scratch using POOD data, so it doesn't need any pretrained weights from outside sources. The original repo does, however, have checkpoint files that can be used for validation.

## 6. Expected Challenges And Risks

- **Dataset preprocessing for Tiny ImageNet and PubFig:** The official codebase comes with CIFAR-10 data pipeline set up right away. To add Tiny ImageNet and PubFig, we will need to change the image resolutions, normalization statistics, data loaders, and maybe even the input layer of the model architecture. PubFig in particular may have images that are not the same size, so you may need to be careful when cropping or resizing them.

- **Hyperparameter sensitivity:** The paper presents findings associated with designated surrogate training epochs, trigger generation cycles, and L-inf radius values. Small changes or differences in random seeds could change how often attacks work, which makes it hard to get the exact same numbers. We will keep track of all the hyperparameters we used and report any differences that we can.

- **Compute and training time:** Running the full pipeline (surrogate warm-up into trigger generation into victim model training into the evaluation) across three datasets and multiple poison ratios for Figure 4 adds up. If GPU access becomes a bottleneck, we will prioritize CIFAR-10 (the most directly supported and documented dataset) and scale back the number of poison ratio sweep points.

- **Reproducibility gaps:** The original code uses fixed random seeds for CIFAR-10, but it might not be able to control all sources of non-determinism (like cuDNN). We anticipate minor numerical discrepancies and will concentrate on whether our results align with a reasonable range of the reported values rather than insisting on precise matches.

## 7. Evaluation And Deliverables

**Metrics:**
- **Attack Success Rate (ASR):** Percentage of triggered test samples (from non target classes) classified as the target class. The paper reports ASR > 98% at 0.05% poison ratio  in that we aim to land within a comparable range.
- **Clean Test Accuracy (CTA):** Accuracy on unmodified test samples, to confirm the backdoor does not degrade normal model performance. Should remain within ~1% of a model trained on clean data to stay consistent with the paper's findings.

**Figures and tables we plan to include:**
- A reproduction of **Table 3**: ASR and CTA across CIFAR-10, Tiny ImageNet, and PubFig at the 0.05% poison ratio, compared side-by-side with the original papers findings.
- A reproduction of **Figure 4**: ASR vs. poison ratio curve on CIFAR-10, plotted against the paper's original curve.
- Visualizations of the generated Narcissus trigger patterns for each dataset.

**Deliverables:**
- Final report (`term-report-Boyington-Khan.pdf`) including all results, analysis, and comparison with the original paper.
- Source code (`term-prog-Boyington-Khan.zip`) containing our adapted Jupyter notebooks and scripts for all three datasets.


## 8. References

- Yi Zeng, Minzhou Pan, Hoang Anh Just, Lingjuan Lyu, Meikang Qiu, and Ruoxi Jia. "Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information." *ACM CCS 2023*. [arXiv:2204.05255](https://arxiv.org/abs/2204.05255)

- Official implementation: [reds-lab/Narcissus](https://github.com/reds-lab/Narcissus)

- CIFAR-10: A. Krizhevsky and G. Hinton. "Learning Multiple Layers of Features from Tiny Images." 2009. Available via `torchvision.datasets.CIFAR10`.

- Tiny ImageNet: A subset of ImageNet (200 classes, 64×64). [https://www.image-net.org/](https://www.image-net.org/)

- PubFig: N. Kumar, A. C. Berg, P. N. Belhumeur, and S. K. Nayar. "Attribute and Simile Classifiers for Face Verification." *ICCV 2009*. [https://www.cs.columbia.edu/CAVE/databases/pubfig/](https://www.cs.columbia.edu/CAVE/databases/pubfig/)
