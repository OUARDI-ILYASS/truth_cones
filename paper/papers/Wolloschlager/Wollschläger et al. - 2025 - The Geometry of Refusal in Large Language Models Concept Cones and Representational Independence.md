# The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence

Tom Wollschlager ¨ \* 1 Jannes Elstner \* 1 Simon Geisler <sup>1</sup> Vincent Cohen-Addad <sup>2</sup> Stephan Gunnemann ¨ <sup>1</sup> Johannes Gasteiger 2 3

# Abstract

The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a *single* refusal direction in the model's activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional *concept cones* that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating the notion of *representational independence* that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs.<sup>1</sup>

# 1. Introduction

The breakthrough of scaling large language models (LLMs) has led to an unprecedented leap in capabilities, driving widespread real-world adoption [\(OpenAI,](#page-9-0) [2022\)](#page-9-0). However, these advancements also introduce serious risks. As artificial intelligence becomes more powerful, it can be misused for harmful purposes, such as attacking critical

\*Equal contribution <sup>1</sup> School of Computation, Information & Technology and Munich Data Science Institute, Technical University of Munich <sup>2</sup>Google Research <sup>3</sup>Now at Anthropic. Correspondence to: Tom Wollschlager ¨ <tom.wollschlaeger@tum.de>.

infrastructure or spreading misinformation. Ensuring that these models remain aligned with human values has become a crucial research challenge [\(Liu et al.,](#page-9-1) [2023;](#page-9-1) [Schwinn et al.,](#page-9-2) [2025\)](#page-9-2). Despite significant progress, LLMs, like all machine learning models, remain vulnerable to adversarial attacks that can bypass alignment mechanisms and induce harmful outputs

<span id="page-0-0"></span>![](_page_0_Picture_10.jpeg)

Figure 1. An example of a 3D concept cone with its basis vectors. All directions in the cone mediate refusal.

[\(Szegedy et al.,](#page-9-3) [2014;](#page-9-3) [Carlini et al.,](#page-8-0) [2024\)](#page-8-0).

Recent work in interpretability has provided valuable insights into how LLMs encode and process information [\(Nanda et al.,](#page-9-4) [2024;](#page-9-4) [Wang et al.,](#page-10-0) [2022;](#page-10-0) [Cunningham et al.,](#page-8-1) [2023;](#page-8-1) [Heinzerling & Inui,](#page-9-5) [2024\)](#page-9-5). Prior studies [\(Bel](#page-8-2)[rose et al.,](#page-8-2) [2023;](#page-8-2) [Gurnee & Tegmark,](#page-9-6) [2023;](#page-9-6) [Marks &](#page-9-7) [Tegmark,](#page-9-7) [2024\)](#page-9-7) suggest that concepts—ranging from simple to complex—are often encoded linearly in the model's residual stream. Methods such as representation engineering [\(Zou et al.,](#page-10-1) [2023a\)](#page-10-1) allow researchers to use input prompts to analyze model behavior by extracting and manipulating such concepts. However, the mechanisms enabling adversarial jailbreaks that bypass alignment safeguards remain poorly understood. Some evidence suggests that refusals to harmful queries are mediated by a single "refusal direction" in activation space [\(Arditi et al.,](#page-8-3) [2024\)](#page-8-3), and that jailbreaks rely on manipulating this direction [\(Yu et al.,](#page-10-2) [2024\)](#page-10-2), yet these assumptions require further examination.

In this work, we go beyond extracting concepts using common input prompt methods by introducing a novel *gradientbased* approach to *representation engineering* which we use to investigate the mechanisms underlying refusal behavior in LLMs. We extract refusal-mediating directions more effectively, improving both precision and control while minimizing unintended side effects, which we demonstrate in

<sup>1</sup> Resources & code: [cs.cit.tum.de/daml/geometry-of-refusal](https://www.cs.cit.tum.de/daml/geometry-of-refusal/)

Section 4. Unlike prior work that assumes model refusal is controlled by a single linear direction, we show in Section 5 that there exist *multi-dimensional polyhedral cones* which contain infinite refusal directions; we show an illustrative example in Figure 1. To further characterize refusal mechanisms in language models, we introduce *representational independence*, a criterion for identifying directions that remain mutually unaffected under intervention, capturing both linear and non-linear dependencies across layers. In Section 6, we demonstrate that even under this strict notion of independence, multiple complementary refusal directions exist.

To summarize, our core **contributions** are:

- We show that our gradient-based representation engineering can advance general LLM understanding and specifically demonstrate its efficacy for understanding refusal mechanisms.
- We introduce representational independence, a practical framework for characterizing how different interventions interact within an LLM's activation space, and use it to find independent refusal directions.
- We show that rather than a single refusal direction, there exist multi-dimensional cones in which all directions mediate refusal.

### <span id="page-1-0"></span>2. Background

#### Notation.

Let  $f: \mathbb{T}^{N_{\text{seq}}} \to \Delta^{N_{\text{seq}} \times |\mathbb{T}|}$  denote a language model, where  $\Delta^{|\mathbb{T}|}$  is the probability simplex over vocabulary  $\mathbb{T}$ . Given a prompt  $p = (t_1, \dots, t_{N_{\text{seq}}}) \in \mathbb{T}^{N_{\text{seq}}}$  consisting of tokens  $t_i$ , each token is first embedded:  $x_i^{(0)} = \text{EMBED}(t_i)$ . The model then processes the token sequence through L layers, where at each layer  $l = 1, \dots, L$  and token position i the following transformation is applied:

$$\tilde{\boldsymbol{x}}_{i}^{(l)} = \boldsymbol{x}_{i}^{(l)} + \text{Attn}^{(l)}(\boldsymbol{x}_{1:i}^{(l)}), \ \ \boldsymbol{x}_{i}^{(l+1)} = \tilde{\boldsymbol{x}}_{i}^{(l)} + \text{MLP}(\tilde{\boldsymbol{x}}_{i}^{(l)})$$

The final residual stream  $x_i^{(L+1)}$  is unembedded to yield logits:  $\ell_i = \text{Unembed}(x_i^{(L+1)})$ . The softmax function converts these logits into a probability distribution over tokens:  $P(t \mid t_1, \dots, t_i) = \text{softmax}(\ell_i)_t$ . We omit technical details that are not critical for this work such as LayerNorm.

Extracting refusal directions. Paired prompts of harmful and harmless requests allow the extraction of a directional feature from the model's residual stream as shown by prior work (Panickssery et al., 2024; Bolukbasi et al., 2016; Burns et al., 2024). Recent studies obtain this direction by computing the *difference-in-means* (DIM) (Panickssery et al., 2024; Arditi et al., 2024; Stolfo et al., 2024) between model representations on datasets of harmful prompts  $\mathcal{D}_{\text{harm}}$  and

harmless prompts  $\mathcal{D}_{good}$ :

$$\boldsymbol{v}_i^{(l)} = \frac{1}{|\mathcal{D}_{\text{harm}}|} \left[ \sum_{p' \in \mathcal{D}_{\text{harm}}} \boldsymbol{x}_i^{(l)}(p') \right] - \frac{1}{|\mathcal{D}_{\text{safe}}|} \left[ \sum_{p \in \mathcal{D}_{\text{safe}}} \boldsymbol{x}_i^{(l)}(p) \right]$$

Here,  $x_i^{(l)}(p)$  represents the residual stream activations at position i, layer l for input prompt p.

**Adversarial steering attacks.** The extracted harmfulness direction can be used to manipulate the model's refusal behavior. With white-box access, an attacker can prompt the model with harmful queries and suppress activations in the harmfulness direction, thereby reducing the model's probability of refusal. This can be done through *directional ablation* of  $\boldsymbol{r}$  (where  $\hat{\boldsymbol{r}}$  denotes the unit vector) (Zou et al., 2023a):

$$\tilde{\boldsymbol{x}}_i^{(l)} = \boldsymbol{x}_i^{(l)} - \hat{\boldsymbol{r}}\hat{\boldsymbol{r}}^{\top}\boldsymbol{x}_i^{(l)},\tag{1}$$

which projects the residual stream to a subspace orthogonal to r, or alternatively through *activation subtraction*:

$$\check{\boldsymbol{x}}_{i}^{(l)} = \boldsymbol{x}_{i}^{(l)} - \alpha \cdot \hat{\boldsymbol{r}}, \tag{2}$$

which subtracts a scaled r from the residual stream. We follow common practice to apply both operations across all token positions and ablation across all layers while doing subtraction only at a single layer.

#### 3. Related Work

Adversarial attacks for LLMs. Many studies have explored hand-crafted adversarial techniques, such as persona modulation (Shah et al., 2023), language modifications (Zhu et al., 2023), or prompt engineering using repetitions and persuasive phrasing (Rao et al., 2024). Other works take a more systematic approach, employing techniques like genetic algorithms and random search (Chen et al., 2024), discrete optimization over input tokens (Zou et al., 2023b), or gradient-based methods to identify high-impact perturbations (Geisler et al., 2024). While identifying these vulnerabilities enables adversarial fine-tuning (Xhonneux et al., 2024) or improved training through Reinforcement Learning with Human Feedback (RLHF), recent works suggest that robustness remains a challenge (Zou et al., 2023a; Schwinn et al., 2024; Geisler et al., 2024; Scholten et al., 2025).

Interpretability of LLMs. A parallel line of research focuses on understanding the internal mechanisms of LLMs, as their natural language outputs provide a unique opportunity to link internal states to interpretable behaviors. Interpretability research has led to the identification of various "features"—concepts represented by distinct activation patterns (Cunningham et al., 2023)—as well as "circuits", which are subnetworks that implement a specific function or behavior. Prominent examples are backup circuits (Nanda et al., 2024) and information mover circuits

(Wang et al., 2022). Many interpretability insights rely on extracting features using paired inputs with opposing semantics (Burns et al., 2024) and then manipulating residual stream activations to elicit specific behaviors (Panickssery et al., 2024). Representation engineering, as proposed by Zou et al. (2023a), investigates the linear representation of concepts such as truthfulness, honesty, and fairness in LLMs. The effectiveness of these methods supports the hypothesis that many features are encoded linearly in LLMs (Marks & Tegmark, 2024). These insights allow researchers to pinpoint and manipulate concept representations or specific circuits, enabling targeted debugging of behaviors, mitigating biases, and advancing safer, more reliable AI systems.

Understanding Refusal Mechanisms. Recent research has focused on understanding the mechanisms underlying refusal behaviors in LLMs. For example, removing safety-critical neurons has been shown to decrease robustness (Wei et al., 2024; Li et al., 2024b). Zheng et al. (2024) demonstrate that adding explicit safety prompts shifts the internal representation along a harmfulness direction. O'Brien et al. (2024) propose to use sparse autoencoders to identify latent features that mediate refusal. The most relevant work to ours is Arditi et al. (2024), which builds on Zou et al. (2023a) and examines the representation of refusal in LLMs. Their work suggests that a single direction a model's activation space determines whether the model accepts or refuses a request. We challenge this notion by showing that refusal is mediated through more nuanced mechanisms.

#### <span id="page-2-0"></span>4. Gradient-based Refusal Directions

Research Question: Can gradient-based representation engineering identify refusal directions?

To investigate the refusal mechanisms in language models, we propose a gradient-based algorithm that identifies directions controlling refusal in the model's activation space. We refer to it as Refusal Direction Optimization (RDO). Unlike prior approaches that extract refusal directions using paired prompts of harmless and harmful instructions (Arditi et al., 2024), our method leverages gradients to find better directions instead of solely relying on model activations. Similar to (Park et al., 2023), we define two key properties for refusal directions:

#### <span id="page-2-2"></span>**Definition 4.1.** Refusal Properties:

- *Monotonic Scaling:* when using the direction for activation addition/subtraction  $\check{\boldsymbol{x}}_i^{(l)} = \boldsymbol{x}_i^{(l)} + \alpha \cdot \boldsymbol{r}$ , the model's probability of refusing instructions should scale monotonically with  $\alpha$ .
- Surgical Ablation: ablating the refusal direction through projection  $\hat{x}_i^{(l)} = x_i^{(l)} \hat{r}\hat{r}^{\top}x_i^{(l)}$  should

#### Algorithm 1 Refusal Direction Optimization (RDO)

<span id="page-2-1"></span>**Input:** Frozen model f, scaling coefficient  $\alpha$ , addition layer index  $l_{\rm add}$ , learning rate  $\eta$ , loss weights  $\lambda_{\rm abl}$ ,  $\lambda_{\rm add}$ ,  $\lambda_{\rm ret}$ , and data  $D = \{(p_{\rm harm,i}, p_{\rm safe,i}, t_{\rm answer,i}, t_{\rm refusal,i}, t_{\rm retain,i})\}_{i=1}^{N}$ . **Output:** Refusal direction  $\boldsymbol{r}$ 

```
1: Initialize r randomly

2: while not converged do

3: Sample batch \mathbf{B} \sim \mathcal{D}

4: \mathcal{L} \leftarrow \mathsf{COMPUTELOSS}(r, f, \mathbf{B})

5: r \leftarrow r - \eta \nabla_r \mathcal{L}

6: r \leftarrow r/||r||_2

7: end while
```

```
1: function COMPUTELOSS(r, f, B)
2: p_{\text{harm}}, p_{\text{safe}}, t_{\text{answer}}, t_{\text{refusal}}, t_{\text{retain}} = B
3: \mathcal{L}_{\text{ablation}} = \text{CELOSS}(f_{\text{ablate}(r)}(p_{\text{harm}}), t_{\text{answer}})
4: \mathcal{L}_{\text{addition}} = \text{CELOSS}(f_{\text{add}}(\alpha \hat{r}, l_{\text{add}})(p_{\text{safe}}), t_{\text{refusal}})
5: \mathcal{L}_{\text{retain}} = \text{KL}(f_{\text{ablate}(r)}(p_{\text{safe}}), f(p_{\text{safe}}), t_{\text{retain}})
6: \mathcal{L} = \lambda_{\text{abl}} \mathcal{L}_{\text{ablation}} + \lambda_{\text{add}} \mathcal{L}_{\text{addition}} + \lambda_{\text{ret}} \mathcal{L}_{\text{retain}}
7: return \mathcal{L}
```

cause the model to answer previously refused harmful prompts, while preserving normal behavior on harmless inputs.

We can encode the desired refusal properties into loss functions, allowing us to find corresponding refusal vectors rusing gradient descent. For the monotonic scaling property, we train the model to refuse harmless instructions  $p_{\text{safe}}$  when running the model f with a modified forward pass  $f_{add(\mathbf{r},l)}$ in which we add r to the activations at layer l. We minimize the cross-entropy between the model output and target refusal response  $t_{refusal}$ . For the surgical ablation property, we similarly compute the cross-entropy between a harmful response target  $t_{answer}$  and the output of a modified forward pass  $f_{\text{ablate}(\boldsymbol{r})}$  to make the model respond to harmful instructions. A key strength of our gradient-based approach is the ability to control any predefined objective and thus we can control the extent to which other concepts are affected during interventions. For this, we use a retain loss based on the Kullback-Leibler (KL) divergence to ensure that directional ablation of r on harmless instructions does not change the model's output over a target response  $t_{\text{retain}}$ . Algorithm 1 shows the full training procedure for our refusal directions.

**Setup.** We construct a dataset of harmless and harmful prompts from the ALPACA (Taori et al., 2023) and SALADBENCH (Li et al., 2024a) datasets (see Appendix A.1). An important consideration for our algorithm is the choice of targets  $t_{\rm answer}$  and  $t_{\rm refusal}$ . Generally, language models differ in their refusal and response styles, which is why we use model-specific targets rather than generating them via uncensored LLMs as in Zou et al. (2024). Specifically, we use

![](_page_3_Figure_1.jpeg)

Figure 2. Attack success rates of refusal directions for different models. We compare the DIM direction baseline that is extracted from prompts against our Refusal Direction Optimization for jailbreaking with directional ablation and activation subtraction.

the DIM refusal direction to generate our targets, though any effective attack can work. For the harmful answers tanswer, we ablate the DIM direction and generate 30 tokens. Similarly, we use activation addition on harmless instructions to produce refusal targets trefusal. For helpful answers on harmless instructions that should be retained tretain, we generate 29 tokens without intervention. The retain loss Lretain is applied over the last 30 tokens, such that the last token of the model's chat template is included. We detail hyperparameters and implementation in Appendix [A.](#page-11-1)

Evaluation. We evaluate our method by training a refusal direction on various models from the Gemma 2 [\(Team et al.,](#page-10-9) [2024\)](#page-10-9), Qwen2.5 [\(Yang et al.,](#page-10-10) [2024\)](#page-10-10), and Llama-3 [\(Dubey](#page-8-8) [et al.,](#page-8-8) [2024\)](#page-8-8) families and compare against the DIM direction for which we use the same setup as [Arditi et al.](#page-8-3) [\(2024\)](#page-8-3) but with our expanded dataset. For a fair comparison, we train the refusal direction at the same layer that the DIM direction is extracted from, and during activation addition/subtraction set the scaling coefficient α to the norm of the DIM direction. We evaluate the jailbreak Attack Success Rate (ASR) on JAILBREAKBENCH [\(Chao et al.,](#page-8-9) [2024\)](#page-8-9) using the STRON-GREJECT fine-tuned judge [\(Souly et al.,](#page-9-19) [2024\)](#page-9-19). For inducing refusal via activation addition, we test 128 harmless instructions sampled from ALPACA using substring matching of common refusal phrases. Model completions for evaluation are generated using greedy decoding with a maximum generation length of 512 tokens.

Does the direction mediate refusal? In Figure [2,](#page-3-0) we show that for jailbreaking, our approach is competitive when using directional ablation and, on average, outperforms DIM when subtracting the refusal direction. Notably, despite not being explicitly optimized for subtraction-based attacks, our direction naturally generalizes to this setting. Figure [9](#page-12-0) shows that adding the refusal direction to harmless inputs induces refusal more effectively with RDO than with DIM, further indicating that our method manipulates refusal more effectively.

<span id="page-3-0"></span>Is the direction more precise? To measure the side effects when intervening with the directions we track benchmark performance. [Arditi et al.](#page-8-3) [\(2024\)](#page-8-3) show that directional ablation with the DIM direction tends to have little impact on benchmark performance, except for TruthfulQA [\(Lin et al.,](#page-9-20) [2021\)](#page-9-20). In Table [1,](#page-3-1) we show that RDO impacts TruthfulQA performance much less severely, reducing the error by 40% on average.

<span id="page-3-1"></span>Table 1. TruthfulQA benchmark performance for directional ablation with the DIM or RDO directions, compared to the baseline (no intervention). Higher values indicate better performance.

| Chat model    | DIM  | RDO (ours)  | Baseline |
|---------------|------|-------------|----------|
| GEMMA 2 2B    | 47.8 | 51.4 (+3.6) | 55.8     |
| GEMMA 2 9B    | 52.8 | 56.7 (+3.9) | 61.1     |
| LLAMA 3 8B    | 48.7 | 51.0 (+2.3) | 52.8     |
| QWEN 2.5 1.5B | 42.9 | 44.0 (+1.1) | 46.5     |
| QWEN 2.5 3B   | 54.2 | 54.5 (+0.3) | 57.2     |
| QWEN 2.5 7B   | 58.7 | 60.0 (+1.3) | 63.1     |
| QWEN 2.5 14B  | 63.3 | 67.9 (+4.6) | 70.8     |

Is our method versatile? Hyperparameter tuning of the retain loss weight λret in Algorithm [1](#page-2-1) allows for balancing between attack success and side effects (see Appendix [B\)](#page-12-1). We observe that for many models—especially those in the Qwen 2.5 family—for the majority of estimated DIM directions, the side-effects are too high, rendering it an unsuccessful attack (see Figure [16\)](#page-16-0). Our method is more flexible than previous work as we can choose the target layer freely while limiting side effects through the retain loss (if possible).

Key Takeaways. Our RDO yields more effective refusal directions with fewer side effects, establishing that gradient-based representation engineering is an effective approach for extracting meaningful directions, while allowing for more modeling freedom such as incorporating side constraints.

![](_page_4_Figure_1.jpeg)

Figure 3. Attack success rate for multi-dimensional cones for Gemma 2, Qwen 2.5 and Llama 3. The cone performance is measured via the performance of Monte Carlo samples which are depicted as boxplot.

### <span id="page-4-0"></span>5. Multi-dimensional Refusal Cones

Research Question: Is refusal in LLMs governed by a single direction, or does it emerge from a more complex underlying geometry?

We extend RDO to higher dimensions by searching for regions in activation space where all vectors control refusal behavior. For this, we optimize an orthonormal basis  $\mathcal{B} = [\boldsymbol{b}_1, \dots, \boldsymbol{b}_N]$  spanning an N-dimensional polyhedral cone  $\mathcal{R}_N = \{\sum_{i=1}^N \lambda_i \boldsymbol{b}_i \mid \lambda_i \geq 0\} \setminus \{\boldsymbol{0}\}$ , where all directions  $\boldsymbol{r} \in \mathcal{R}_N$  satisfy the refusal properties (Definition 4.1). Since all directions in the cone correspond to the same refusal concept, we also refer to this as a *concept cone*. The constraint  $\lambda_i \geq 0$  ensures that all directions within the cone consistently strengthen refusal behavior. Without this constraint, allowing negative coefficients could introduce opposing effects, reducing the overall effectiveness. Enforcing orthogonality of the basis vectors prevents finding co-linear directions. Note that in practice, directions in activation space cannot be scaled arbitrarily high without model degeneration, which effectively bounds  $\lambda_i$ .

In Algorithm 2, we describe the procedure to find the cone's basis vectors. The basis vectors are initialized randomly and iteratively optimized using projected gradient descent. We compute the previous losses defined in Algorithm 1 on Monte Carlo samples from the cone, as well as on the basis vectors themselves. Computing the loss on the basis vectors improves both stability and the lower bounds of the ASR. This is because the basis vectors are the boundaries of the cone and thus tend to degrade first. After each step, we project the basis back onto the cone using the

#### **Algorithm 2** Refusal Cone Optimization (RCO)

- <span id="page-4-2"></span><span id="page-4-1"></span>1: **Initialize**  $\mathcal{B} = [\boldsymbol{b}_1, \dots, \boldsymbol{b}_n]$  randomly
- 2: while not converged do
- 3: Sample batch  $B \sim D$
- 4:  $\mathcal{L}_{\text{sample}} \leftarrow \mathbb{E}_{r \sim \text{Sample}(\mathcal{B})}[\text{ComputeLoss}(r, f, B)]$
- 5:  $\mathcal{L}_{\text{basis}} \leftarrow \frac{1}{n} \sum_{i=1}^{n} \text{COMPUTELOSS}(\boldsymbol{b}_i, f, \mathbf{B})$
- 6:  $\mathcal{L} = \mathcal{L}_{sample} + \mathcal{L}_{basis}$
- 7:  $\mathcal{B} \leftarrow \mathcal{B} \eta \nabla_{\mathcal{B}} \mathcal{L}$
- 8:  $\mathcal{B} \leftarrow \mathsf{GRAMSCHMIDT}(\mathcal{B})$
- 9: end while
- 1: **function** SAMPLE( $\mathcal{B}$ )
- 2:  $s \sim \text{Unif}(x \in \mathbb{R}^n_+ : ||x||_2 = 1)$
- 3:  $r = \mathcal{B}s$
- 4: **return** *r*

Gram-Schmidt orthogonalization procedure. Because the directional ablation operation uses the normalized  $\hat{r}$  rather than r, sampling convex combinations of the basis vectors and normalizing them would introduce a bias towards the basis vectors themselves. Instead, we sample unit vectors in the cone uniformly to ensure better coverage of the space.

Can we find refusal concept cones? We train cones of increasing dimensionality using the same experimental setup as described in Section 4. We measure the cone's effectiveness in mediating refusal by sampling 256 vectors from each cone and computing the ASRs of the samples for directional ablation. We show the results in Figure 3 and confirm that the directions in the cones have the desired refusal properties in Figure 14. Notably, we identify refusal-mediating cones with dimensions up to five across all tested models. This suggests that the activation space in language models exhibits a general property where refusal behavior is en-

![](_page_5_Figure_1.jpeg)

Figure 4. Refusal evaluation for different cone dimensions for the Qwen2.5 model family. The cone performance for models with fewer parameters degrades faster with increasing cone dimension compared to larger models.

coded within multi-dimensional cones rather than a single linear direction.

#### Do larger models contain higher-dimensional cones?

In Figure [4,](#page-5-0) we evaluate the effect of model size within the Qwen 2.5 family. We observe that across all model sizes, the lower bounds of cone performance degrade significantly as dimensionality increases. In other words, a higher number of sampled directions have low ASR. Larger models appear to support higher-dimensional refusal cones. A plausible explanation is that models with larger residual stream dimensions (e.g., 5120 for the 14B model vs. 1536 for the 1.5B model) allow for more distinct and orthogonal directions that mediate refusal. Finally, in Figure [11,](#page-13-0) we confirm that directions sampled from these cones effectively induce refusal behavior, further supporting the notion that multiple axes contribute to the model's refusal decision.

#### Do different directions uniquely influence refusal?

To further investigate the role of different vectors, we assess whether multiple sampled cone directions influence the model in complementary ways. Specifically, we sample varying numbers of directions from Gemma-2-2B's fourdimensional refusal cone and, for each prompt, select the most effective one under directional ablation (more details in Appendix [A\)](#page-11-1). To ensure a fair comparison, we use temperature sampling with the single-dimension RDO direction to generate the same number of attacks and similarly select the most effective instance. We study Gemma 2 2B and sample from its four-dimensional cone, since performance degrades significantly for larger dimensions (see Figure [10\)](#page-13-1).

Figure [5](#page-5-1) shows that sampling multiple directions leads to higher ASR compared to sampling with various temperatures in the low-sample regime. For a higher number of

<span id="page-5-0"></span>samples, the randomness dominates the success of the attack. However, the higher ASR in the low-sample regime suggests that different directions capture distinct, complementary aspects of the refusal mechanism. Additionally, Figure [13](#page-15-1) reveals that ASR increases with cone dimensionality but plateaus at four dimensions. This trend indicates that higherdimensional cones offer an advantage over single-direction manipulation, likely by influencing complementary mechanisms. The plateau likely occurs because the model does not support higher-dimensional refusal cones.

![](_page_5_Figure_10.jpeg)

<span id="page-5-1"></span>Figure 5. ASR for best-of-N sampling using N samples from the 4-dimensional refusal cone of Gemma-2-2B, compared to best-of-N sampling with temperature T using the single-dimension RDO.

Key Takeaways. We show that refusal mechanisms in LLMs span high-dimensional polyhedral cones, capturing diverse aspects of refusal behavior. This highlights their geometric complexity and demonstrates the effectiveness of our gradient-based method in identifying intricate structures.

### <span id="page-6-0"></span>6. Mechanistic Understanding of Directions

Research Question: Are there genuinely independent directions that influence a model's refusal behavior? Can we access the discovered refusal directions through perturbations in the token space?

In the previous section, we demonstrated that refusal behavior spans a multi-dimensional cone with infinitely many directions. However, whether the orthogonal refusal-mediating basis vectors manipulate independent mechanisms remains an open question. In this section, we conduct a mechanistic analysis to investigate how these directions interact within the model's activation space and whether they can be directly influenced through input manipulation. This allows us to determine whether they are merely latent properties of the network or actively utilized by the model in response to specific prompts.

#### 6.1. Representational Independence

We defined the basis vectors of the cones to be orthogonal, which is often considered an indicator of causal independence. The intuition is that if two vectors are orthogonal, they each influence a third vector without interfering with the other. Mathematically, for the directions r, v and representation  $x_i^{(l)}$  we have:

$$\text{if } \boldsymbol{r}^T\boldsymbol{v} = 0 \text{ then } \boldsymbol{r}^T(\boldsymbol{x}_i^{(l)} - \boldsymbol{v}\boldsymbol{v}^T\boldsymbol{x}_i^{(l)}) = \boldsymbol{r}^T\boldsymbol{x}_i^{(l)}.$$

However, despite this mathematical property, recent work by Park et al. (2024) suggests that in language models, conclusions about causal independence cannot be drawn using orthogonality measured with the Euclidean scalar product. Although their assumptions differ from ours, especially since they assume a one-to-one mapping from output feature to direction in activation space, their experiments suggest that independent directions are almost orthogonal. This motivates a deeper empirical examination of how orthogonal refusal directions in language models interact in practice.

Are orthogonal directions independent? To explore this, we first use RDO to identify a direction  $\boldsymbol{r}$  for Gemma 2 2B that is orthogonal to the DIM direction  $\boldsymbol{v}$ , i.e.,  $\boldsymbol{r}^{\top}\boldsymbol{v}=0$ . We then measure how much one direction is influenced when ablating the other direction by monitoring the cosine similarity  $\cos(\lambda,\mu) = \frac{\lambda^{\top}\mu}{||\lambda||\cdot||\mu||}$  between the prompt's representation in the residual stream  $\boldsymbol{x}$  and the directions  $\boldsymbol{v}$  and  $\boldsymbol{r}$ . Specifically, we track:  $\cos(\boldsymbol{r},\boldsymbol{x}_i^{(l)}(p_{\text{harm}}))$  and  $\cos(\boldsymbol{v},\boldsymbol{x}_i^{(l)}(p_{\text{harm}}))$  at the last token position and for all layers  $l \in \{0,\ldots,L\}$  on 128 harmful instructions in our validation set. Intuitively, ablating a causally independent direction in earlier layers should not intervene with the reference direction in later layers. Otherwise, there is some indirect influence through the non-linear transformations of the neural network.

![](_page_6_Figure_9.jpeg)

<span id="page-6-1"></span>Figure 6. Influence of representational independence. Figure (a) shows the cosine similarity between  $RDO_{\perp}$ , a refusal direction orthogonal to DIM, and the model activations in a normal forward pass (solid line) compared to a forward pass where DIM is ablated (striped line). Figure (b) shows the reverse scenario. In Figure (c) and (d) we contrast how the DIM direction and a representationally independent direction (RepInd) influence each other.

The top row of Figure 6 shows how the cosine similarity between the RDO and DIM directions changes under intervention. The left plot shows the cosine similarity between the RDO direction and the activations on a normal forward pass (solid line) and while ablating the DIM direction (dashed line). The right plot presents the reverse setting. Despite enforced orthogonality, ablating RDO indirectly reduces the representation of the DIM direction in the model activations in the later layers, as measured by cosine similarity. This effect is reciprocal, suggesting that orthogonality alone does not guarantee independence throughout the network.

Motivated by this observation, we introduce a stricter notion of independence: *Representational Independence (RepInd)*:

**Definition 6.1.** The directions  $\lambda, \mu \in \mathbb{R}^d$  are representationally independent (under directional ablation) with respect to the activations x of a model in a set of layers  $l \in L$  if:

$$\begin{split} \forall l \in L : \cos(\boldsymbol{x}^{(l)}, \lambda) &= \cos(\tilde{\boldsymbol{x}}^{(l)}_{abl(\mu)}, \lambda) \\ \text{and } \cos(\boldsymbol{x}^{(l)}, \mu) &= \cos(\tilde{\boldsymbol{x}}^{(l)}_{abl(\lambda)}, \mu). \end{split}$$

This means that, instead of relying solely on orthogonality, we define two directions as representationally independent if ablating one has no effect on how much the other is represented in the model activations. To enforce this property, we extend Algorithm 1 with an additional loss term that penalizes changes in cosine similarity at the last token position

![](_page_7_Figure_1.jpeg)

Figure 7. Performance of RepInd directions. Each direction is representationally independent to all previous directions and the DIM direction.

when ablating on harmful instructions:

$$\begin{split} \mathcal{L}_{\text{RepInd}} &= \frac{1}{|L|} \sum_{l \in L} \left[ \left( \cos(x^{(l)}, \boldsymbol{r}) - \cos(\tilde{x}_{abl(\boldsymbol{v})}^{(l)}, \boldsymbol{r}) \right)^2 \right. \\ &\left. + \left( \cos(x^{(l)}, \boldsymbol{v}) - \cos(\tilde{x}_{abl(\boldsymbol{r})}^{(l)}, \boldsymbol{v}) \right)^2 \right]. \end{split}$$

Do independent directions exist? With this extension, we can find a direction that is RepInd from the DIM direction, yet still fulfills the refusal properties from Definition [4.1.](#page-2-2) We show the representational independence in the second row of Figure [6,](#page-6-1) where we see that the RepInd and DIM direction barely affect each other's representation under directional ablation.

We iteratively search for additional directions that are not only RepInd to DIM but also of all previously identified RepInd directions. Despite these strong constraints, we successfully identify at least five such directions that maintain an ASR significantly above random vector intervention (Figure [7\)](#page-7-0), as well as a refusal cone with RepInd basis vectors (Figure [12\)](#page-14-0). However, in Figure [7](#page-7-0) and Figure [12](#page-14-0) performance degrades more rapidly compared to the results in Section [4](#page-2-0) and Section [5.](#page-4-0) This decline could be attributed to the increased difficulty of the optimization problem due to additional constraints. Alternatively, it may suggest that Gemma 2 2B possesses a limited number of directions that independently contribute to refusal. If the latter is true, this implies that the directions in the refusal cones exhibit nonlinear dependencies. Nevertheless, these results show that refusal in LLMs is mediated by multiple *independent* mechanisms, underpinning the idea that refusal behavior is more nuanced than previously assumed.

#### 6.2. Manipulation from input

Can we access these directions from the input? Having found several independent directions that are distinct from DIM, we investigate whether these directions can ever be "used" by the model, by checking if they are accessible from the input or if they live in regions that no combination of input tokens activates. To this end, we use GCG [\(Zou et al.,](#page-10-4) [2023b\)](#page-10-4) to train adversarial suffixes, which are extensions to the prompts that aim to circumvent the safety alignment. In addition to the standard cross-entropy loss on an affirmative target, we add a loss term that incentivizes the suffix to ablate RepInd 1.

<span id="page-7-0"></span>![](_page_7_Figure_10.jpeg)

<span id="page-7-1"></span>Figure 8. Representation of RepInd 1 in model activations on harmful instructions before and after adversarial attacks with GCG.

In Figure [8,](#page-7-1) we show the cosine similarities between RepInd 1 and the model activations on both harmful prompts pharm from JAILBREAK-BENCH and the same prompts with adversarial suffixes padv. We observe that GCG is able to create suffixes that significantly reduce how much RepInd 1 is represented. These suffixes successfully jail-

break the model 36% of the time, which is similar to the ASR of RepInd 1.

Key Takeaways. We demonstrate the ability to identify independent refusal directions, revealing that these directions correspond to distinct underlying concepts and can be directly accessed through input manipulations. This further underscores the utility of our representational independence framework, which provides a generalizable approach for analyzing and understanding a wide range of representational interventions in LLMs.

### 7. Limitations

While our work provides new insights into the geometry of refusal in LLMs, some limitations remain. The refusal directions we compute are all optimized on the same targets, which may limit their ability to capture fully distinct mechanisms. Extending our method to incorporate diverse targets or leveraging reinforcement learning with a judge-based reward function could help identify additional independent mechanisms [\(Geisler et al.,](#page-8-10) [2025\)](#page-8-10). Furthermore, while we establish the existence of higher-dimensional refusal cones, we cannot rule out the possibility of other yet-undiscovered regions in the model that mediate refusal.

### 8. Conclusion

This work advances the understanding of refusal mechanisms in LLMs by introducing gradient-based representation engineering as a powerful tool for identifying and analyzing refusal directions. Our method yields more effective

refusal directions with fewer side effects, demonstrating its viability for extracting meaningful structures while allowing for greater modeling flexibility. We establish that refusal behaviors can be better understood via high-dimensional polyhedral cones in activation space rather than a single linear direction, highlighting their complex spatial structures. Additionally, we introduce representational independence and show that within this space of independent directions multiple refusal directions exist and correspond to distinct mechanisms. Our gradient-based representation engineering approach can be extended to identify various concepts beyond refusal by simply changing the optimization targets. The generated findings provide new insights into the geometry of aligned LLMs, highlighting the importance of structured, gradient-based approaches in LLM interpretability and safety.

## Acknowledgements

This project was conducted in collaboration with and supported by funding from Google Research. We thank Dominik Fuchsgruber and Leo Schwinn for feedback on an early version of the manuscript.

### Impact Statement

Understanding how refusal mechanisms in language models work could potentially aid adversaries in developing more effective attacks. However, our research aims to deepen the understanding of refusal mechanisms to help the community develop more robust and reliable safety systems. By focusing on open-source models requiring white-box access, our findings are primarily applicable to improving defensive capabilities rather than compromising deployed systems. We believe the positive impact of advancing model alignment and safety through better theoretical understanding outweighs the potential risks, making this research valuable to share with the research community.

# References

- <span id="page-8-3"></span>Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N. Refusal in language models is mediated by a single direction, 2024. URL <https://arxiv.org/abs/2406.11717>.
- <span id="page-8-2"></span>Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., and Biderman, S. Leace: Perfect linear concept erasure in closed form, 2023. URL [https:](https://arxiv.org/abs/2306.03819) [//arxiv.org/abs/2306.03819](https://arxiv.org/abs/2306.03819).
- <span id="page-8-4"></span>Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., and Kalai, A. Man is to computer programmer as woman is to homemaker? debiasing word embeddings, 2016. URL <https://arxiv.org/abs/1607.06520>.

- <span id="page-8-5"></span>Burns, C., Ye, H., Klein, D., and Steinhardt, J. Discovering latent knowledge in language models without supervision, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2212.03827) [2212.03827](https://arxiv.org/abs/2212.03827).
- <span id="page-8-0"></span>Carlini, N., Nasr, M., Choquette-Choo, C. A., Jagielski, M., Gao, I., Awadalla, A., Koh, P. W., Ippolito, D., Lee, K., Tramer, F., and Schmidt, L. Are aligned neural networks adversarially aligned?, 2024. URL [https://arxiv.](https://arxiv.org/abs/2306.15447) [org/abs/2306.15447](https://arxiv.org/abs/2306.15447).
- <span id="page-8-9"></span>Chao, P., Debenedetti, E., Robey, A., Andriushchenko, M., Croce, F., Sehwag, V., Dobriban, E., Flammarion, N., Pappas, G. J., Tramer, F., Hassani, H., and Wong, E. ` Jailbreakbench: An open robustness benchmark for jailbreaking large language models. In *NeurIPS Datasets and Benchmarks Track*, 2024.
- <span id="page-8-6"></span>Chen, Z., Zhu, J., and Chen, A. *Eliciting Offesnive Responses from Large Language Models: A Genetic Algorithm*. Springer, 2024.
- <span id="page-8-1"></span>Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse Autoencoders Find Highly Interpretable Features in Language Models, October 2023. URL [http://arxiv.org/abs/2309.](http://arxiv.org/abs/2309.08600) [08600](http://arxiv.org/abs/2309.08600). arXiv:2309.08600 [cs].
- <span id="page-8-8"></span>Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.
- <span id="page-8-11"></span>Fiotto-Kaufman, J., Loftus, A. R., Todd, E., Brinkmann, J., Juang, C., Pal, K., Rager, C., Mueller, A., Marks, S., Sharma, A. S., et al. Nnsight and ndif: Democratizing access to foundation model internals. *arXiv preprint arXiv:2407.14561*, 2024.
- <span id="page-8-12"></span>Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac'h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for few-shot language model evaluation, 07 2024. URL [https://zenodo.org/records/](https://zenodo.org/records/12608602) [12608602](https://zenodo.org/records/12608602).
- <span id="page-8-7"></span>Geisler, S., Wollschlager, T., Abdalla, M. H. I., Gasteiger, ¨ J., and Gunnemann, S. Attacking Large Lan- ¨ guage Models with Projected Gradient Descent, February 2024. URL [http://arxiv.org/abs/2402.](http://arxiv.org/abs/2402.09154) [09154](http://arxiv.org/abs/2402.09154). arXiv:2402.09154 [cs].
- <span id="page-8-10"></span>Geisler, S., Wollschlager, T., Abdalla, M. H. I., Gasteiger, ¨ J., and Gunnemann, S. Reinforce adversarial attacks on ¨ large language models: An adaptive, distributional, and semantic objective, February 2025.

- <span id="page-9-6"></span>Gurnee, W. and Tegmark, M. Language models represent space and time. *arXiv preprint arXiv:2310.02207*, 2023.
- <span id="page-9-5"></span>Heinzerling, B. and Inui, K. Monotonic representation of numeric properties in language models. *arXiv preprint arXiv:2403.10381*, 2024.
- <span id="page-9-18"></span>Li, L., Dong, B., Wang, R., Hu, X., Zuo, W., Lin, D., Qiao, Y., and Shao, J. Salad-bench: A hierarchical and comprehensive safety benchmark for large language models. *arXiv preprint arXiv:2402.05044*, 2024a.
- <span id="page-9-14"></span>Li, T., Wang, Z., Liu, W., Wu, M., Dou, S., Lv, C., Wang, X., Zheng, X., and Huang, X. Revisiting jailbreaking for large language models: A representation engineering perspective, 2024b. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2401.06824) [2401.06824](https://arxiv.org/abs/2401.06824).
- <span id="page-9-20"></span>Lin, S., Hilton, J., and Evans, O. Truthfulqa: Measuring how models mimic human falsehoods. *arXiv preprint arXiv:2109.07958*, 2021.
- <span id="page-9-22"></span>Lin, Z., Wang, Z., Tong, Y., Wang, Y., Guo, Y., Wang, Y., and Shang, J. Toxicchat: Unveiling hidden challenges of toxicity detection in real-world user-ai conversation. *arXiv preprint arXiv:2310.17389*, 2023.
- <span id="page-9-1"></span>Liu, Y., Yao, Y., Ton, J.-F., Zhang, X., Cheng, R. G. H., Klochkov, Y., Taufiq, M. F., and Li, H. Trustworthy llms: A survey and guideline for evaluating large language models' alignment. *arXiv preprint arXiv:2308.05374*, 2023.
- <span id="page-9-7"></span>Marks, S. and Tegmark, M. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, 2024. URL [https://](https://arxiv.org/abs/2310.06824) [arxiv.org/abs/2310.06824](https://arxiv.org/abs/2310.06824).
- <span id="page-9-4"></span>Nanda, N., Olah, C., Olsson, C., Elhage, N., and Hume, T. Attribution patching: Activation patching at industrial scale. [https://www.neelnanda.](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) [io/mechanistic-interpretability/](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) [attribution-patching](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching), 2024. Accessed: 2025-01-10.
- <span id="page-9-15"></span>O'Brien, K., Majercak, D., Fernandes, X., Edgar, R., Chen, J., Nori, H., Carignan, D., Horvitz, E., and Poursabzi-Sangde, F. Steering language model refusal with sparse autoencoders, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2411.11296) [abs/2411.11296](https://arxiv.org/abs/2411.11296).
- <span id="page-9-0"></span>OpenAI. Introducing chatgpt, November 2022. URL <https://openai.com/blog/chatgpt/>. Accessed: 2025-01-26.
- <span id="page-9-8"></span>Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., and Turner, A. M. Steering Llama 2 via Contrastive Activation Addition, July 2024. URL [http://arxiv.](http://arxiv.org/abs/2312.06681) [org/abs/2312.06681](http://arxiv.org/abs/2312.06681). arXiv:2312.06681 [cs].

- <span id="page-9-16"></span>Park, K., Choe, Y. J., and Veitch, V. The linear representation hypothesis and the geometry of large language models. *arXiv preprint arXiv:2311.03658*, 2023.
- <span id="page-9-21"></span>Park, K., Choe, Y. J., and Veitch, V. The Linear Representation Hypothesis and the Geometry of Large Language Models, July 2024. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2311.03658) [2311.03658](http://arxiv.org/abs/2311.03658). arXiv:2311.03658 [cs].
- <span id="page-9-11"></span>Rao, A., Vashistha, S., Naik, A., Aditya, S., and Choudhury, M. Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks, March 2024. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2305.14965) [2305.14965](http://arxiv.org/abs/2305.14965). arXiv:2305.14965 [cs].
- <span id="page-9-13"></span>Scholten, Y., Gunnemann, S., and Schwinn, L. A proba- ¨ bilistic perspective on unlearning and alignment for large language models. In *The Thirteenth International Conference on Learning Representations*, 2025.
- <span id="page-9-12"></span>Schwinn, L., Dobre, D., Xhonneux, S., Gidel, G., and Gunnemann, S. Soft prompt threats: Attacking safety ¨ alignment and unlearning in open-source LLMs through the embedding space. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024.
- <span id="page-9-2"></span>Schwinn, L., Scholten, Y., Wollschlager, T., Xhonneux, ¨ S., Casper, S., Gunnemann, S., and Gidel, G. Ad- ¨ versarial alignment for llms requires simpler, reproducible, and more measurable objectives. *arXiv preprint arXiv:2502.11910*, 2025.
- <span id="page-9-10"></span>Shah, R., Feuillade-Montixi, Q., Pour, S., Tagade, A., Casper, S., and Rando, J. Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation, November 2023. URL [http://arxiv.](http://arxiv.org/abs/2311.03348) [org/abs/2311.03348](http://arxiv.org/abs/2311.03348). arXiv:2311.03348 [cs].
- <span id="page-9-19"></span>Souly, A., Lu, Q., Bowen, D., Trinh, T., Hsieh, E., Pandey, S., Abbeel, P., Svegliato, J., Emmons, S., Watkins, O., and Toyer, S. A strongreject for empty jailbreaks, 2024.
- <span id="page-9-9"></span>Stolfo, A., Balachandran, V., Yousefi, S., Horvitz, E., and Nushi, B. Improving instruction-following in language models through activation steering, 2024. URL [https:](https://arxiv.org/abs/2410.12877) [//arxiv.org/abs/2410.12877](https://arxiv.org/abs/2410.12877).
- <span id="page-9-3"></span>Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., and Fergus, R. Intriguing properties of neural networks, 2014. URL [https://arxiv.org/](https://arxiv.org/abs/1312.6199) [abs/1312.6199](https://arxiv.org/abs/1312.6199).
- <span id="page-9-17"></span>Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford alpaca: An instruction-following llama model. [https://github.com/tatsu-lab/](https://github.com/tatsu-lab/stanford_alpaca) [stanford\\_alpaca](https://github.com/tatsu-lab/stanford_alpaca), 2023.

- <span id="page-10-9"></span>Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Rame, A., et al. Gemma 2: Improving open ´ language models at a practical size. *arXiv preprint arXiv:2408.00118*, 2024.
- <span id="page-10-0"></span>Wang, K., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small, November 2022. URL [http://arxiv.org/abs/2211.](http://arxiv.org/abs/2211.00593) [00593](http://arxiv.org/abs/2211.00593). arXiv:2211.00593 [cs].
- <span id="page-10-11"></span>Wang, W., Tu, Z., Chen, C., Yuan, Y., Huang, J.-t., Jiao, W., and Lyu, M. R. All languages matter: On the multilingual safety of large language models. *arXiv preprint arXiv:2310.00905*, 2023.
- <span id="page-10-6"></span>Wei, B., Huang, K., Huang, Y., Xie, T., Qi, X., Xia, M., Mittal, P., Wang, M., and Henderson, P. Assessing the brittleness of safety alignment via pruning and low-rank modifications, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2402.05162) [abs/2402.05162](https://arxiv.org/abs/2402.05162).
- <span id="page-10-5"></span>Xhonneux, S., Sordoni, A., Gunnemann, S., Gidel, G., and ¨ Schwinn, L. Efficient adversarial training in LLMs with continuous attacks. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024.
- <span id="page-10-10"></span>Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5 technical report. *arXiv preprint arXiv:2412.15115*, 2024.
- <span id="page-10-2"></span>Yu, L., Do, V., Hambardzumyan, K., and Cancedda, N. Robust llm safeguarding via refusal feature adversarial training. *arXiv preprint arXiv:2409.20089*, 2024.
- <span id="page-10-7"></span>Zheng, C., Yin, F., Zhou, H., Meng, F., Zhou, J., Chang, K.-W., Huang, M., and Peng, N. On Prompt-Driven Safeguarding for Large Language Models, June 2024. URL [http://arxiv.org/abs/2401.](http://arxiv.org/abs/2401.18018) [18018](http://arxiv.org/abs/2401.18018). arXiv:2401.18018 [cs].
- <span id="page-10-3"></span>Zhu, S., Zhang, R., An, B., Wu, G., Barrow, J., Wang, Z., Huang, F., Nenkova, A., and Sun, T. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models, December 2023. URL [http://](http://arxiv.org/abs/2310.15140) [arxiv.org/abs/2310.15140](http://arxiv.org/abs/2310.15140). arXiv:2310.15140 [cs].
- <span id="page-10-1"></span>Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.- K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., and Hendrycks, D. Representation Engineering: A Top-Down Approach to AI Transparency, October 2023a. URL [http://arxiv.org/abs/2310.](http://arxiv.org/abs/2310.01405) [01405](http://arxiv.org/abs/2310.01405). arXiv:2310.01405 [cs].

- <span id="page-10-4"></span>Zou, A., Wang, Z., Kolter, J. Z., and Fredrikson, M. Universal and Transferable Adversarial Attacks on Aligned Language Models, July 2023b. URL [http://arxiv.](http://arxiv.org/abs/2307.15043) [org/abs/2307.15043](http://arxiv.org/abs/2307.15043). arXiv:2307.15043 [cs].
- <span id="page-10-8"></span>Zou, A., Phan, L., Wang, J., Duenas, D., Lin, M., Andriushchenko, M., Kolter, J. Z., Fredrikson, M., and Hendrycks, D. Improving alignment and robustness with circuit breakers. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024.

### <span id="page-11-1"></span>A. Setup Details

#### <span id="page-11-0"></span>A.1. Datasets

We construct our experimental dataset using harmful and harmless instructions from established benchmarks. For harmful instructions, we draw from SALADBENCH [\(Li et al.,](#page-9-18) [2024a\)](#page-9-18), a comprehensive collection of adversarial prompts from diverse sources. We exclude the Multilingual [\(Wang et al.,](#page-10-11) [2023\)](#page-10-11) and ToxicChat [\(Lin et al.,](#page-9-22) [2023\)](#page-9-22) sources since they are unsuited as harmful instructions. Afterwards, we sample up to 256 instructions from each remaining source. This results in 1,184 instructions for training and 128 for validation. We sample equal numbers of harmless instructions from the ALPACA dataset, and additionally reserve 128 more harmless instructions for testing.

#### A.2. Models

We exclusively use chat models for our experiments, but omit "IT" and "INSTRUCT" from model names. We use each chat model's default chat template throughout our analysis.

| Table 2. Model families, sizes, and references. |  |  |
|-------------------------------------------------|--|--|
|-------------------------------------------------|--|--|

| Model family     | Sizes             | Reference           |
|------------------|-------------------|---------------------|
| QWEN2.5 INSTRUCT | 1.5B, 3B, 7B, 14B | Yang et al. (2024)  |
| GEMMA 2 IT       | 2B, 9B            | Team et al. (2024)  |
| LLAMA-3 INSTRUCT | 8B                | Dubey et al. (2024) |

### <span id="page-11-2"></span>A.3. Hyperparameters and Implementation

Table 3. Hyperparameters for all algorithms

| Component            | Parameter                     | Value                        |
|----------------------|-------------------------------|------------------------------|
| Training             | Total Batch Size              | 16                           |
|                      | Gradient Accumulation Steps   | 16                           |
|                      | Base Learning Rate            | 0.01                         |
|                      | Learning Rate Reduction       | Every 5 batches if plateaued |
|                      | Learning Rate Factor          | Divide by 1/10 up to 2 times |
|                      | Optimizer                     | AdamW                        |
|                      | Weight Decay                  | 0                            |
| Main Loss            | Ablation Loss Weight λabl     | 1.0                          |
|                      | Addition Loss Weight λadd     | 0.2                          |
|                      | Retain Loss Weight λret       | 1.0                          |
| Monte Carlo Sampling | Samples per Accumulation Step | 16                           |
|                      | Effective Samples per Batch   | 256                          |
| RepInd               | RepInd Loss Weight λind       | 200                          |
|                      | Layer Cutoff                  | 0.9                          |

Table [3](#page-11-2) presents the hyperparameters used in our algorithms. Since our method converges before completing a full epoch, we do not utilize validation scores during training. Instead, after convergence, we apply the direction selection algorithm from [Arditi et al.](#page-8-3) [\(2024\)](#page-8-3) to identify the optimal refusal direction from the last 20 training steps.

Implementation and Evaluation Framework. All algorithms and exploratory experiments are implemented using the NNsight [\(Fiotto-Kaufman et al.,](#page-8-11) [2024\)](#page-8-11) library. Additionally, we use the LM Evaluation Harness [\(Gao et al.,](#page-8-12) [2024\)](#page-8-12) to run TruthfulQA [\(Lin et al.,](#page-9-20) [2021\)](#page-9-20) with default settings, with the exception that we enable the use of each model's default chat templates.

Retain and Representational Independence Loss Computation. The retain loss is computed as the KL divergence between the probability distributions derived from the logits of the model with and without directional ablation, masked

![](_page_12_Figure_1.jpeg)

Figure 9. Refusal scores of different models on harmless instructions after activation addition that aims to induce refusal.

over a target response and the last token of the chat template. The resulting value is then averaged across tokens. For a single instruction psafe with its target tretain, we formalize the loss as follows:

<span id="page-12-0"></span>
$$\mathcal{L}_{\text{retain}} = \text{KL}(f_{\text{ablate}(\boldsymbol{r})}(p_{\text{safe}}), f(p_{\text{safe}}), t_{\text{retain}}) = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \sum_{t \in \mathbb{T}} f(p_{\text{safe}} + t_{\text{retain}})_{i,t} \log \frac{f(p_{\text{safe}} + t_{\text{retain}})_{i,t}}{f_{\text{ablate}}(p_{\text{safe}} + t_{\text{retain}})_{i,t}},$$

where I contains the target token indexes and the last instruction token's index, the subscript i, t denotes the model output at sequence position i and vocabulary index t as defined in Section [2,](#page-1-0) and + denotes concatenation.

For the implementation of the representational independence loss, LRepInd, we compute the average loss over the tokens in the harmful instructions pharm. The RepInd loss is computed over the first 90% of layers, as applying it too close to the unembedding layer overly constrains the model's output.

Selection of Refusal and Independent Directions In Algorithm [1,](#page-2-1) after training the refusal directions to convergence, we again use the direction selection algorithm from [Arditi et al.](#page-8-3) [\(2024\)](#page-8-3) to identify the most effective directions from the final 20 training steps.

In Section [5,](#page-4-0) we extend this selection process to determine a basis where all basis vectors effectively mediate refusal (from the last 20 bases of the training). If no such basis exists, we instead select the basis where the samples are most effective for directional ablation using the refusal score heuristic from the selection algorithm.

Training Procedure for Representational Independence Directions In Section [6,](#page-6-0) our approach to training and validating representationally independent (RepInd) directions differs because of high variance between different runs. For each RepInd direction, we train five candidate vectors and select the one with the lowest refusal score on our validation set. This process is repeated five times, ultimately producing our final set of RepInd directions. The RepInd loss is computed as the sum of losses over all vectors that the current vector should remain independent of.

### <span id="page-12-1"></span>B. Additional Experiments

In this section, we present further experimental results. Figure [9](#page-12-0) demonstrates that adding the refusal direction successfully induces refusal behavior across all models for both DIM and RDO. Similarly, Figure [10](#page-13-1) illustrates the refusal cone performance for Gemma 2, confirming the existence of higher-dimensional refusal cones within the Gemma 2 family. The results suggest that the maximum cone dimensionality may be four, as the lower bounds of the ASR drop sharply beyond this point. In Figure [11](#page-13-0) we apply refusal cones to various Qwen 2.5 models across different dimensions, revealing that inducing refusal is significantly easier than conducting an attack. Notably, most directions even in high-dimensional cones remain effective at inducing refusal responses.

Figure [13](#page-15-1) examines the attack success rate when sampling multiple vectors from various N-dimensional refusal cones and selecting the best-performing sample per prompt for Gemma 2, 2B. We observe that ASR improves with increasing cone dimensionality but plateaus at four dimensions, suggesting that higher-dimensional cones provide an advantage over single-direction manipulation by capturing complementary mechanisms. The plateau likely results from the model's inability

<span id="page-13-1"></span>![](_page_13_Figure_1.jpeg)

Figure 10. Attack success rates in refusal cones of different dimensions for the Gemma 2 model family. We observe that for the Gemma 2 2B the lower bounds start to degrade significantly for dimension 5.

![](_page_13_Figure_3.jpeg)

<span id="page-13-0"></span>Figure 11. Using refusal cones to induce refusal across various Qwen 2.5 models with different dimensions. We observe that inducing refusal is generally easier than executing an attack. In this setting, nearly all dimensions maintain strong performance in eliciting refusal responses, even for benign requests.

<span id="page-14-0"></span>![](_page_14_Figure_1.jpeg)

Figure 12. Attack success rates in refusal cones of different dimensions for Gemma 2 2B where the basis vectors are trained to be representationally independent.

to encode higher-dimensional refusal cones, a hypothesis further supported by Figure [10.](#page-13-1)

Moving on to the ablation study, Figure [15](#page-16-1) presents an analysis of the relationship between the retain loss weight and model performance on the Qwen 2.5 3B model. The left plot illustrates the performance under directional ablation and activation subtraction, with results averaged over five runs per loss weight. For this model, a loss weight of 1 or lower yields the best generalization, while increasing the penalty for unintended side effects on harmless instructions significantly degrades performance. On the right, we examine how the retain loss weights generalize to the validation KL score. This allows us to abstract from specific training conditions and evaluate how effectively the loss weights transfer beyond the training setup.

Finally, we assess the performance of the baseline across different (layer, token) combinations. Figure [16](#page-16-0) visualizes the effectiveness of the direction selection algorithm from [Arditi et al.](#page-8-3) [\(2024\)](#page-8-3) for DIM directions in the Qwen 2.5 7B model. Among the evaluated token and layer pairs, only one direction is found to be effective for both inducing refusal through activation addition and maintaining low side effects. Transparent data points indicate (layer, token) combinations that were filtered out due to their inability to induce refusal reliably. Additionally, the red line represents the KL-divergence threshold, used to estimate potential side effects of directional ablation on harmless instructions.

![](_page_15_Figure_1.jpeg)

Figure 13. Attack success rates when sampling vectors from the N-dimensional refusal cones and selecting the best-performing sample per prompt for Gemma 2 2B. ASR increases with cone dimensionality but plateaus at four dimensions, suggesting that higher-dimensional cones provide an advantage over single-direction manipulation by capturing complementary mechanisms. The plateau likely arises because Algorithm [2](#page-4-1) cannot find an additional basis vector that preserves the refusal properties in the cone, suggesting that the model does not support a cone of this dimension. Figure [10](#page-13-1) also provides evidence for this claim.

<span id="page-15-1"></span>![](_page_15_Figure_3.jpeg)

<span id="page-15-0"></span>Figure 14. Refusal scores of refusal vectors sampled from Gemma 2 2B refusal cones compared to the DIM direction when scaling the norm of the added direction α for the activation addition intervention. The refusal score is the heuristic from [Arditi et al.](#page-8-3) [\(2024\)](#page-8-3) here, and we compute it on 64 harmful validation instructions, with mean and standard deviation over 64 samples per alpha.

![](_page_16_Figure_1.jpeg)

Figure 15. The left plot shows the relationship between the retain loss weight and performance when using the trained direction for directional ablation and activation subtraction on the Qwen 2.5 3B model, with mean and standard deviation over 5 runs per loss weight. For this specific model, a loss weight of 1 or smaller results in the best generalization, and performance degrades significantly as the direction is penalized more for unintended side-effects on harmless instructions. On the right, we also show performance depending on how the directions generalized to the validation KL-score.

<span id="page-16-1"></span><span id="page-16-0"></span>![](_page_16_Figure_3.jpeg)

Figure 16. Analysis of the selection direction algorithm from [Arditi et al.](#page-8-3) [\(2024\)](#page-8-3) for the DIM directions of Qwen 2.5 7B. Among the token and layer combinations, only a single direction is identified as viable for both inducing refusal via activation addition and having low side-effects. Transparent points represent (layer, token) pairs that are filtered out because of ineffectiveness in inducing refusal. The red line indicates the KL-divergence threshold used to estimate potential side-effects of directional ablation on harmless instructions.

### C. Assets

In the following, we show the licenses for all the assets we used in this work: different models from Table [4](#page-17-0) and the datasets that we use for evaluation and training; see Table [5.](#page-17-1)

#### <span id="page-17-0"></span>C.1. Models

Table 4. The list of models used in this work.

| Model                    | Source              | Accessed via | License                        |
|--------------------------|---------------------|--------------|--------------------------------|
| Qwen 2.5 {1.5B, 7B, 14B} | Yang et al. (2024)  | Link         | Apache 2.0 License             |
| Qwen 2.5 {3B}            | Yang et al. (2024)  | Link         | Qwen Research License          |
| Gemma 2 2B               | Team et al. (2024)  | Link         | Apache 2.0 License             |
| Gemma 2 9B               | Team et al. (2024)  | Link         | Gemma Terms of Use             |
| Llama-3 8B               | Dubey et al. (2024) | Link         | Meta Llama 3 Community License |
| StrongREJECT Judge       | Souly et al. (2024) | Link         | MIT License                    |

### <span id="page-17-1"></span>C.2. Datasets

Table 5. The list of datasets used in this work.

| Dataset        | Source              | Accessed via | License            |
|----------------|---------------------|--------------|--------------------|
| SALADBENCH     | Li et al. (2024a)   | Link         | Apache License 2.0 |
| ALPACA         | Taori et al. (2023) | Link         | Apache License 2.0 |
| JAILBREAKBENCH | Chao et al. (2024)  | Link         | MIT License        |
| TRUTHFULQA     | Lin et al. (2021)   | Link         | Apache License 2.0 |