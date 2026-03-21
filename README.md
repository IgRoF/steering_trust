# Steering Trust

A research agenda on AI honesty, deception and trustworthiness. This repo tracks my work on evaluating and steering honesty in large language models, starting with benchmark replication and building towards mechanistic interpretability and honesty interventions.

> **Status (21 March 2026):** I'm currently working on Step 1 of this agenda: replicating and extending the [MASK honesty benchmark](https://arxiv.org/abs/2503.03750) on 2026 frontier models, as part of [BlueDot Impact](https://bluedot.org)'s Technical AI Safety Project Sprint. See [Current Project](#current-project-mask-replication--extension) below for details and preliminary results. The original steering vectors work described in earlier versions of this README is planned for a later phase; I decided to start with the benchmarking foundations first.

---

## Research Agenda

My long-term research interest is investigating whether we can detect a model's internal representation of its own honesty, find a related steering vector for this trait, and apply it to maximize honesty and minimize deception across different situations and benchmarks. If we can be reasonably certain that a model is being mostly honest, many (but not all) alignment problems could be minimized.

This breaks down into roughly three phases:

1. **Evaluating honesty at the frontier:** Replicate the MASK benchmark on the latest frontier models to establish baselines and test whether the original finding (that honesty does not improve with scale) persists. *(in progress)*
2. **Investigating alternative elicitation methods:** Explore whether dishonesty can be elicited via methods other than prompt pressure (e.g. activation steering, contrastive prompts), to disentangle genuine deceptiveness from instruction-following compliance / role-laying. *(planned)*
3. **Steering for honesty:** Apply and compare honesty interventions (LoRRA, contrastive activation addition, SAEs, weight steering) and measure their impact on MASK and other benchmarks. *(planned)*

---

## Current Project: MASK Replication & Extension

**Context:** The [MASK benchmark](https://www.mask-benchmark.ai/) (Ren et al., 2025) is the first large-scale evaluation that disentangles model honesty from accuracy. The original paper found that frontier LLMs lie 20–60% of the time when pressured, and that honesty does not improve with training compute. These results were based on 27 models with training FLOPs up to ~10²⁶.

**What I'm doing:** Replicating MASK on 9 current models from 7 families (GPT, Claude, Gemini, DeepSeek, Grok, Llama, Qwen) using the [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/mask) framework. This extends the original FLOP range into ~10²⁶–10²⁷ with models released after the paper, while overlapping with the paper's upper range for continuity.

**Progress so far:**
- Environment and workflow set up (Windows / local laptop for API models, RunPod for open-weight models)
- Provider SDKs installed and tested for OpenAI, Anthropic, Google, DeepSeek, and xAI
- Preliminary n=10 pilot runs completed for 6 API models, confirming the pipeline works
- Phase 1 de-risking effectively complete; moving into full 1000-record runs

Results and a write-up will be added here as they become available.

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
