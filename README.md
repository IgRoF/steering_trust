# Steering Trust

A research agenda on AI honesty, deception and trustworthiness. This repo tracks my work on evaluating and steering honesty in large language models, starting with benchmark replication and building towards mechanistic interpretability and honesty interventions.

> **➡️ Latest work (11 April 2026):** The first public version of my **[MASK Benchmark Replication](mask-benchmark-replication/)** package is now available in this repo. If you are here to look at concrete results, that subfolder is the right place to start: it contains the full replication write-up, the dated result tables, the reproducibility guides, and a curated public script surface. See [mask-benchmark-replication/README.md](mask-benchmark-replication/README.md) for the entry point, and [mask-benchmark-replication/docs/results/results-overview.md](mask-benchmark-replication/docs/results/results-overview.md) for the shortest summary of the current findings.

---

## Research Agenda

My long-term research interest is investigating whether we can detect a model's internal representation of its own honesty, find a related steering vector for this trait, and apply it to maximize honesty and minimize deception across different situations and benchmarks. If we can be reasonably certain that a model is being mostly honest, many (but not all) alignment problems could be minimized.

This breaks down into roughly three phases:

1. **Evaluating honesty at the frontier:** Replicate the MASK benchmark on the latest frontier models to establish baselines and test whether the original finding (that honesty does not improve with scale) persists. *(first public version released)*
2. **Investigating alternative elicitation methods:** Explore whether dishonesty can be elicited via methods other than prompt pressure (e.g. activation steering, contrastive prompts), to disentangle genuine deceptiveness from instruction-following compliance / role-laying. *(planned)*
3. **Steering for honesty:** Apply and compare honesty interventions (LoRRA, contrastive activation addition, SAEs, weight steering) and measure their impact on MASK and other benchmarks. *(planned)*

---

## Current Project: MASK Replication & Extension

**Status (11 April 2026):** The first public version of the replication package has been uploaded under [`mask-benchmark-replication/`](mask-benchmark-replication/). It covers nine completed full runs (1,000 samples each) on current frontier models from seven families, plus the supporting reproducibility guides and dated result tables. Phase 1 of the agenda is now in a state where it can be read and rerun by others.
 
**Brief summary of findings:**
 
- The broad MASK pattern still holds: several frontier models remain factually accurate while still lying often under pressure.
- Claude Opus 4.6 is the strongest honesty result in the completed set (0.838 honesty, 0.906 accuracy), and GPT-5.4 is notably above the paper's GPT-4.5-preview row on both axes.
- DeepSeek V3.2 and Grok 4.20 still occupy the high-accuracy, low-honesty corner of the comparison set, showing the same "knows but lies" pattern as their earlier model from the original paper.
- Gemini Flash looks mixed: more honest than Gemini 2.0 Flash in the paper, but less accurate on the first completed Gemini full run.
- The GPT-4o overlap rerun and the latest Llama 3.1 OpenRouter rerun act as paper-comparable anchors, and both land close to the paper's appendix values.
 
For the full tables, the family-evolution comparison, the accuracy comparability notes, and the reproducibility guides, please go to [mask-benchmark-replication/](mask-benchmark-replication/).

---

## About Me

I'm Ignacio, an Aerospace Engineer transitioning towards a career in technical AI safety research. I am particularly interested in questions of mechanistic interpretability and evaluations of honesty and deception in LLMs.

I'm currently building my research portfolio through programs like BlueDot Impact's Technical AI Safety Project Sprint, and I plan to apply the skills I'm developing here towards more exhaustive interpretability work in the near future.

I have two goals for this project: the main one is learning-by-doing. I want to gain hands-on experience with empirical AI safety research, from running large-scale LLM evaluations to implementing interpretability and steering techniques on open-weight models.

My second goal is to explore some questions that interest me about the topics of AI honesty, truthfulness and deception. These might have been already investigated elsewhere, but I want to try my hand at them first before doing a deeper dive into the literature. I’m currently focused on identifying and eliciting honesty in frontier models as a key aspect of AI safety. My Theory of Change for this is, very simplified:

1. For models to be accurate and effective, they must have some kind of internal representation of the world.
2. If a model is being deceptive, there must be some mismatch between its internal world model and their outputs/actions.
3. This mismatch should be, in principle, traceable somewhere in the hidden states of the models.
4. If deception can be found, it can theoretically be steered away from.
5. If deception is minimized, many misalignment issues, such as scheming, can also be reduced.

This is a very idealized picture, but I believe any progress on this front would be helpful as a basis for other safety and alignment interventions.

---

## Related Reading

Papers and resources that inform this research agenda (non-exhaustive):

- [The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems](https://arxiv.org/abs/2503.03750) (Ren et al., Mar-2025)
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) (Zou et al., Oct-2023)
- [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248) (Turner et al., Aug-2023)
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (Panickssery et al., Dec-2023)
- [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509) (Chen et al., Jul-2025)
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (Arditi et al., Jun-2024)
- [Emergent Introspective Awareness in Large Language Models](https://transformer-circuits.pub/2025/introspection) (Lindsey et al., Oct-2025)

---

*This repo is a work in progress. I will update it regularly as results come in and the research agenda evolves.*
