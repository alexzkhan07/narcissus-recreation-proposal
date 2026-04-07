# Recreation of Narcissus Clean-Label Backdoor Attack

## Team

- Max Boyington
- Alex Khan

## 1. Project Choice

NARCISSUS: A Practical Clean-Label Backdoor Attack with Limited Information


NARCISSUS is a clean-label backdoor attack, meaning the attack uses maliciously crafted data injected into a dataset without changing its labels (hence "clean-label"). This paper's attack was highly efficient, creating backdoors in models using only 0.05% of the training data. By optimizing the trigger to match a target class's internal features, it evades modern defenses and remains effective across a variety of datasets and real-world situations. 


Repository: [reds-lab/Narcissus](https://github.com/reds-lab/Narcissus)  
Paper: [arXiv:2204.05255](https://arxiv.org/abs/2204.05255)

## 2. Specific Task You Propose To Recreate

- State exactly what part of the original project you plan to reproduce.
- Be specific about the experiment, claim, model, dataset, or result you will target first.

### What to include here

- Which result or experiment you are recreating
- What counts as the main success criterion
- Whether this is a pure reproduction or a reproduction plus a small extension

## 3. Goals

- State the concrete goals of the project.
- Focus on what you want to demonstrate by the end of the term.

### What to include here

- Primary goal
- Secondary goals
- What you hope to learn or verify

## 4. Plan To Achieve The Goals

- Outline the steps you will take to complete the project.
- Keep this practical and sequential.

### What to include here

- Environment setup
- Source code review
- Dataset acquisition or preprocessing
- Running the baseline experiment
- Evaluating results
- Writing up findings

## 5. Required Resources

- List the main resources needed to carry out the recreation.

### What to include here

- Datasets
- Compute requirements
- Libraries or frameworks
- Any pretrained models, checkpoints, or external tools

## 6. Expected Challenges And Risks

- Briefly note what might make the recreation difficult.

### What to include here

- Dataset access issues
- Long training time
- Hardware constraints
- Reproducibility gaps in the original code or paper

## 7. Evaluation And Deliverables

- Explain how you will judge whether the project was successful.
- State what you expect to turn in or present.

### What to include here

- Metrics you will report
- Figures, tables, or plots you plan to include
- Final code, report, notebook, or presentation deliverables

## 8. References

- Add the paper citation.
- Add the repository link.
- Add any dataset or benchmark links that are central to the project.
