# **The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence** 

**Tom Wollschl¨ager**[* 1] **Jannes Elstner**[* 1] **Simon Geisler**[1] **Vincent Cohen-Addad**[2] **Stephan G¨unnemann**[1] **Johannes Gasteiger**[2 3] 

## **Abstract** 

The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a _single_ refusal direction in the model’s activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional _concept cones_ that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating 

the notion of _representational independence_ that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs.[1] 

## **1. Introduction** 

The breakthrough of scaling large language models (LLMs) has led to an unprecedented leap in capabilities, driving widespread real-world adoption (OpenAI, 2022). However, these advancements also introduce serious risks. As artificial intelligence becomes more powerful, it can be misused for harmful purposes, such as attacking critical 

*Equal contribution 1School of Computation, Information & Technology and Munich Data Science Institute, Technical University of Munich[2] Google Research[3] Now at Anthropic. Correspondence to: Tom Wollschl¨ager _<_ tom.wollschlaeger@tum.de _>_ . 

> 1 Resources & code: cs.cit.tum.de/daml/geometry-of-refusal 

infrastructure or spreading misinformation. En7s Cone — > Basis vector suring that these models remain aligned with human values has become a crucial research challenge (Liu et al., 2023; Schwinn et al., 2025). Despite significant progress, LLMs, like all machine learning models, remain vulnera- _Figure 1._ An example of a 3D ble to adversarial attacks concept cone with its basis vectors. All directions in the cone that can bypass alignmediate refusal. ment mechanisms and induce harmful outputs (Szegedy et al., 2014; Carlini et al., 2024). 

Recent work in interpretability has provided valuable insights into how LLMs encode and process information (Nanda et al., 2024; Wang et al., 2022; Cunningham et al., 2023; Heinzerling & Inui, 2024). Prior studies (Belrose et al., 2023; Gurnee & Tegmark, 2023; Marks & Tegmark, 2024) suggest that concepts—ranging from simple to complex—are often encoded linearly in the model’s residual stream. Methods such as representation engineering (Zou et al., 2023a) allow researchers to use input prompts to analyze model behavior by extracting and manipulating such concepts. However, the mechanisms enabling adversarial jailbreaks that bypass alignment safeguards remain poorly understood. Some evidence suggests that refusals to harmful queries are mediated by a single “refusal direction” in activation space (Arditi et al., 2024), and that jailbreaks rely on manipulating this direction (Yu et al., 2024), yet these assumptions require further examination. 

In this work, we go beyond extracting concepts using common input prompt methods by introducing a novel _gradientbased_ approach to _representation engineering_ which we use to investigate the mechanisms underlying refusal behavior in LLMs. We extract refusal-mediating directions more effectively, improving both precision and control while minimizing unintended side effects, which we demonstrate in 

1 

**The Geometry of Refusal in Large Language Models** 

Section 4. Unlike prior work that assumes model refusal is controlled by a single linear direction, we show in Section 5 that there exist _multi-dimensional polyhedral cones_ which contain infinite refusal directions; we show an illustrative example in Figure 1. To further characterize refusal mechanisms in language models, we introduce _representational independence_ , a criterion for identifying directions that remain mutually unaffected under intervention, capturing both linear and non-linear dependencies across layers. In Section 6, we demonstrate that even under this strict notion of independence, multiple complementary refusal directions exist. 

To summarize, our core **contributions** are: 

- We show that our gradient-based representation engineering can advance general LLM understanding and specifically demonstrate its efficacy for understanding refusal mechanisms. 

- We introduce representational independence, a practical framework for characterizing how different interventions interact within an LLM’s activation space, and use it to find independent refusal directions. 

- We show that rather than a single refusal direction, there exist multi-dimensional cones in which all directions mediate refusal. 

## **2. Background** 

## **Notation.** 

Let _f_ : T _[N]_[seq] _→_ ∆ _[N]_[seq] _[×|]_[T] _[|]_ denote a language model, where ∆ _[|]_[T] _[|]_ is the probability simplex over vocabulary T. Given a prompt _p_ = ( _t_ 1 _, . . . , tN_ seq ) _∈_ T _[N]_[seq] consisting of tokens _ti_ , each token is first embedded: _x_[(0)] _i_ = EMBED( _ti_ ). The model then processes the token sequence through _L_ layers, where at each layer _l_ = 1 _, . . . , L_ and token position _i_ the following transformation is applied: 

**==> picture [235 x 14] intentionally omitted <==**

The final residual stream _x_[(] _i[L]_[+1)] is unembedded to yield logits: _ℓi_ = UNEMBED( _x_[(] _i[L]_[+1)] ). The softmax function converts these logits into a probability distribution over tokens: _P_ ( _t | t_ 1 _, . . . , ti_ ) = softmax( _ℓi_ ) _t_ . We omit technical details that are not critical for this work such as LayerNorm. 

**Extracting refusal directions.** Paired prompts of harmful and harmless requests allow the extraction of a directional feature from the model’s residual stream as shown by prior work (Panickssery et al., 2024; Bolukbasi et al., 2016; Burns et al., 2024). Recent studies obtain this direction by computing the _difference-in-means_ (DIM) (Panickssery et al., 2024; Arditi et al., 2024; Stolfo et al., 2024) between model representations on datasets of harmful prompts _D_ harm and 

harmless prompts _D_ good: 

**==> picture [242 x 37] intentionally omitted <==**

Here, _**x**_[(] _i[l]_[)][(] _[p]_[)][ represents the residual stream activations at] position _i_ , layer _l_ for input prompt _p_ . 

**Adversarial steering attacks.** The extracted harmfulness direction can be used to manipulate the model’s refusal behavior. With white-box access, an attacker can prompt the model with harmful queries and suppress activations in the harmfulness direction, thereby reducing the model’s probability of refusal. This can be done through _directional ablation_ of _**r**_ (where _**r**_ ˆ denotes the unit vector) (Zou et al., 2023a): 

**==> picture [166 x 14] intentionally omitted <==**

which projects the residual stream to a subspace orthogonal to _**r**_ , or alternatively through _activation subtraction_ : 

**==> picture [159 x 14] intentionally omitted <==**

which subtracts a scaled _**r**_ from the residual stream.We follow common practice to apply both operations across all token positions and ablation across all layers while doing subtraction only at a single layer. 

## **3. Related Work** 

**Adversarial attacks for LLMs.** Many studies have explored hand-crafted adversarial techniques, such as persona modulation (Shah et al., 2023), language modifications (Zhu et al., 2023), or prompt engineering using repetitions and persuasive phrasing (Rao et al., 2024). Other works take a more systematic approach, employing techniques like genetic algorithms and random search (Chen et al., 2024), discrete optimization over input tokens (Zou et al., 2023b), or gradient-based methods to identify high-impact perturbations (Geisler et al., 2024). While identifying these vulnerabilities enables adversarial fine-tuning (Xhonneux et al., 2024) or improved training through Reinforcement Learning with Human Feedback (RLHF), recent works suggest that robustness remains a challenge (Zou et al., 2023a; Schwinn et al., 2024; Geisler et al., 2024; Scholten et al., 2025). 

**Interpretability of LLMs.** A parallel line of research focuses on understanding the internal mechanisms of LLMs, as their natural language outputs provide a unique opportunity to link internal states to interpretable behaviors. Interpretability research has led to the identification of various “features”—concepts represented by distinct activation patterns (Cunningham et al., 2023)—as well as “circuits”, which are subnetworks that implement a specific function or behavior. Prominent examples are backup circuits (Nanda et al., 2024) and information mover circuits 

2 

**The Geometry of Refusal in Large Language Models** 

(Wang et al., 2022). Many interpretability insights rely on extracting features using paired inputs with opposing semantics (Burns et al., 2024) and then manipulating residual stream activations to elicit specific behaviors (Panickssery et al., 2024). Representation engineering, as proposed by Zou et al. (2023a), investigates the linear representation of concepts such as truthfulness, honesty, and fairness in LLMs. The effectiveness of these methods supports the hypothesis that many features are encoded linearly in LLMs (Marks & Tegmark, 2024). These insights allow researchers to pinpoint and manipulate concept representations or specific circuits, enabling targeted debugging of behaviors, mitigating biases, and advancing safer, more reliable AI systems. 

**Understanding Refusal Mechanisms.** Recent research has focused on understanding the mechanisms underlying refusal behaviors in LLMs. For example, removing safetycritical neurons has been shown to decrease robustness (Wei et al., 2024; Li et al., 2024b). Zheng et al. (2024) demonstrate that adding explicit safety prompts shifts the internal representation along a harmfulness direction. O’Brien et al. (2024) propose to use sparse autoencoders to identify latent features that mediate refusal. The most relevant work to ours is Arditi et al. (2024), which builds on Zou et al. (2023a) and examines the representation of refusal in LLMs. Their work suggests that a single direction a model’s activation space determines whether the model accepts or refuses a request. We challenge this notion by showing that refusal is mediated through more nuanced mechanisms. 

## **4. Gradient-based Refusal Directions** 

Research Question: Can gradient-based representation engineering identify refusal directions? 

To investigate the refusal mechanisms in language models, we propose a gradient-based algorithm that identifies directions controlling refusal in the model’s activation space. We refer to it as Refusal Direction Optimization _(_ RDO _)_ . Unlike prior approaches that extract refusal directions using paired prompts of harmless and harmful instructions (Arditi et al., 2024), our method leverages gradients to find better directions instead of solely relying on model activations. Similar to (Park et al., 2023), we define two key properties for refusal directions: 

**Definition 4.1.** Refusal Properties: 

- _Monotonic Scaling:_ when using the direction for activation addition/subtraction _**x**_ ˇ[(] _i[l]_[)] = _**x**_[(] _i[l]_[)] + _α ·_ _**r**_ , the model’s probability of refusing instructions should scale monotonically with _α_ . 

- _Surgical Ablation:_ ablating the refusal direction ˜ ˆˆ 

- through projection _**x**_[(] _i[l]_[)] = _**x**_[(] _i[l]_[)] _−_ _**rr**[⊤]_ _**x**_[(] _i[l]_[)] should 

**Algorithm 1** Refusal Direction Optimization (RDO) 

**Input:** Frozen model _f_ , scaling coefficient _α_ , addition layer index _l_ add, learning rate _η_ , loss weights _λ_ abl, _λ_ add, _λ_ ret, and data _D_ = _{_ ( _p_ harm _,i, p_ safe _,i, t_ answer _,i, t_ refusal _,i, t_ retain _,i_ ) _}[N] i_ =1[.] **Output:** Refusal direction _**r**_ 

|1: <br>2: <br>3:<br>4:<br>5:<br>6:<br>7:|**Initialize**_r_randomly<br> **while**not converged**do**<br>Sample batch B_∼D_<br>_L ←_COMPUTELOSS(**_r_**_, f,_B)<br>**_r_** _←_**_r_**_−η∇_**_r_**_L_<br>**_r_** _←_**_r_**_/||_**_r_**_||_2<br> **end while**|
|---|---|



|1:|**function**COMPUTELOSS(**_r_**_, f,_B)|
|---|---|
|2:|_p_harm_, p_safe_, t_answer_, t_refusal_, t_retain =_B_|
|3:|_L_ablation = CELOSS(_f_ablate(**_r_**)(_p_harm)_, t_answer)|
|4:|_L_addition = CELOSS(_f_add(_α_ˆ**_r_**_,l_add)(_p_safe)_, t_refusal)|
|5:|_L_retain =KL(_f_ablate(**_r_**)(_p_safe)_, f_(_p_safe)_, t_retain)|
|6:|_L_=_λ_abl_L_ablation+_λ_add_L_addition+_λ_ret_L_retain|
|7:|**return**_L_|



- cause the model to answer previously refused harmful prompts, while preserving normal behavior on harmless inputs. 

We can encode the desired refusal properties into loss functions, allowing us to find corresponding refusal vectors _**r**_ using gradient descent. For the monotonic scaling property, we train the model to refuse harmless instructions _p_ safe when running the model _f_ with a modified forward pass _f_ add( _**r** ,l_ ) in which we add _**r**_ to the activations at layer _l_ . We minimize the cross-entropy between the model output and target refusal response _t_ refusal. For the surgical ablation property, we similarly compute the cross-entropy between a harmful response target _t_ answer and the output of a modified forward pass _f_ ablate( _**r**_ ) to make the model respond to harmful instructions. A key strength of our gradient-based approach is the ability to control any predefined objective and thus we can control the extent to which other concepts are affected during interventions. For this, we use a retain loss based on the Kullback-Leibler (KL) divergence to ensure that directional ablation of _**r**_ on harmless instructions does not change the model’s output over a target response _t_ retain. Algorithm 1 shows the full training procedure for our refusal directions. 

**Setup.** We construct a dataset of harmless and harmful prompts from the ALPACA (Taori et al., 2023) and SALADBENCH (Li et al., 2024a) datasets (see Appendix A.1). An important consideration for our algorithm is the choice of targets _t_ answer and _t_ refusal. Generally, language models differ in their refusal and response styles, which is why we use model-specific targets rather than generating them via uncensored LLMs as in Zou et al. (2024). Specifically, we use 

3 

**The Geometry of Refusal in Large Language Models** 

_Figure 2._ Attack success rates of refusal directions for different models. We compare the DIM direction baseline that is extracted from prompts against our Refusal Direction Optimization for jailbreaking with directional ablation and activation subtraction. 

the DIM refusal direction to generate our targets, though any effective attack can work. For the harmful answers _t_ answer, we ablate the DIM direction and generate 30 tokens. Similarly, we use activation addition on harmless instructions to produce refusal targets _t_ refusal. For helpful answers on harmless instructions that should be retained _t_ retain, we generate 29 tokens without intervention. The retain loss _L_ retain is applied over the last 30 tokens, such that the last token of the model’s chat template is included. We detail hyperparameters and implementation in Appendix A. 

**Evaluation.** We evaluate our method by training a refusal direction on various models from the Gemma 2 (Team et al., 2024), Qwen2.5 (Yang et al., 2024), and Llama-3 (Dubey et al., 2024) families and compare against the DIM direction for which we use the same setup as Arditi et al. (2024) but with our expanded dataset. For a fair comparison, we train the refusal direction at the same layer that the DIM direction is extracted from, and during activation addition/subtraction set the scaling coefficient _α_ to the norm of the DIM direction. We evaluate the jailbreak Attack Success Rate (ASR) on JAILBREAKBENCH (Chao et al., 2024) using the STRONGREJECT fine-tuned judge (Souly et al., 2024). For inducing refusal via activation addition, we test 128 harmless instructions sampled from ALPACA using substring matching of common refusal phrases. Model completions for evaluation are generated using greedy decoding with a maximum generation length of 512 tokens. 

**Does the direction mediate refusal?** In Figure 2, we show that for jailbreaking, our approach is competitive when using directional ablation and, on average, outperforms DIM when subtracting the refusal direction. Notably, despite not being explicitly optimized for subtraction-based attacks, our direction naturally generalizes to this setting. Figure 9 shows that adding the refusal direction to harmless inputs induces refusal more effectively with RDO than with DIM, further indicating that our method manipulates refusal more effectively. 

**Is the direction more precise?** To measure the side effects when intervening with the directions we track benchmark performance. Arditi et al. (2024) show that directional ablation with the DIM direction tends to have little impact on benchmark performance, except for TruthfulQA (Lin et al., 2021). In Table 1, we show that RDO impacts TruthfulQA performance much less severely, reducing the error by 40% on average. 

_Table 1._ TruthfulQA benchmark performance for directional ablation with the DIM or RDO directions, compared to the baseline (no intervention). Higher values indicate better performance. 

|Chat model|DIM|RDO**(ours)**|Baseline|
|---|---|---|---|
|GEMMA2 2B|47.8|51.4(+3.6)|55.8|
|GEMMA2 9B|52.8|56.7(+3.9)|61.1|
|LLAMA3 8B|48.7|51.0(+2.3)|52.8|
|QWEN2.5 1.5B|42.9|44.0(+1.1)|46.5|
|QWEN2.5 3B|54.2|54.5(+0.3)|57.2|
|QWEN2.5 7B|58.7|60.0(+1.3)|63.1|
|QWEN2.5 14B|63.3|67.9(+4.6)|70.8|



**Is our method versatile?** Hyperparameter tuning of the retain loss weight _λ_ ret in Algorithm 1 allows for balancing between attack success and side effects (see Appendix B). We observe that for many models—especially those in the Qwen 2.5 family—for the majority of estimated DIM directions, the side-effects are too high, rendering it an unsuccessful attack (see Figure 16). Our method is more flexible than previous work as we can choose the target layer freely while limiting side effects through the retain loss (if possible). 

**Key Takeaways.** Our RDO yields more effective refusal directions with fewer side effects, establishing that gradient-based representation engineering is an effective approach for extracting meaningful directions, while allowing for more modeling freedom such as incorporating side constraints. 

4 

**The Geometry of Refusal in Large Language Models** 

_Figure 3._ Attack success rate for multi-dimensional cones for Gemma 2, Qwen 2.5 and Llama 3. The cone performance is measured via the performance of Monte Carlo samples which are depicted as boxplot. 

## **5. Multi-dimensional Refusal Cones** 

Research Question: Is refusal in LLMs governed by a single direction, or does it emerge from a more complex underlying geometry? 

We extend RDO to higher dimensions by searching for regions in activation space where all vectors control refusal behavior. For this, we optimize an orthonormal basis _B_ = [ _**b**_ 1 _, . . . ,_ _**b** N_ ] spanning an _N_ -dimensional polyhedral _N_ cone _RN_ = _{ i_ =1 _[λ][i]_ _**[b]**[i][|][λ][i][≥]_[0] _[}\{]_ **[0]** _[}]_[, where all direc-] tions _**r** ∈RN_ satisfy the refusal properties (Definition 4.1). Since all directions in the cone correspond to the same refusal concept, we also refer to this as a _concept cone_ . The constraint _λi ≥_ 0 ensures that all directions within the cone consistently strengthen refusal behavior. Without this constraint, allowing negative coefficients could introduce opposing effects, reducing the overall effectiveness. Enforcing orthogonality of the basis vectors prevents finding co-linear directions. Note that in practice, directions in activation space cannot be scaled arbitrarily high without model degeneration, which effectively bounds _λi_ . 

In Algorithm 2, we describe the procedure to find the cone’s basis vectors. The basis vectors are initialized randomly and iteratively optimized using projected gradient descent. We compute the previous losses defined in Algorithm 1 on Monte Carlo samples from the cone, as well as on the basis vectors themselves. Computing the loss on the basis vectors improves both stability and the lower bounds of the ASR. This is because the basis vectors are the boundaries of the cone and thus tend to degrade first. After each step, we project the basis back onto the cone using the 

**Algorithm 2** Refusal Cone Optimization (RCO) 

- 1: **Initialize** _B_ = [ _**b**_ 1 _, . . . ,_ _**b** n_ ] randomly 2: **while** not converged **do** 3: Sample batch B _∼D_ 4: _L_ sample _←_ E _**r** ∼_ Sample( _B_ )[COMPUTELOSS( _**r** , f,_ B)] 5: _L_ basis _← n_[1] _ni_ =1[C][OMPUTE][L][OSS][(] _**[b]**[i][, f,]_[ B][)] 6: _L_ = _L_ sample + _L_ basis 7: _B ←B − η∇BL_ 8: _B ←_ GRAMSCHMIDT( _B_ ) 9: **end while** 

- 1: **function** SAMPLE( _B_ ) 2: _**s** ∼_ Unif( _**x** ∈_ R _[n]_ +[:] _[ ||]_ _**[x]**[||]_[2][= 1)] 3: _**r**_ = _B_ _**s**_ 4: **return** _**r**_ 

Gram-Schmidt orthogonalization procedure. Because the directional ablation operation uses the normalized _**r**_ ˆ rather than _**r**_ , sampling convex combinations of the basis vectors and normalizing them would introduce a bias towards the basis vectors themselves. Instead, we sample unit vectors in the cone uniformly to ensure better coverage of the space. 

**Can we find refusal concept cones?** We train cones of increasing dimensionality using the same experimental setup as described in Section 4. We measure the cone’s effectiveness in mediating refusal by sampling 256 vectors from each cone and computing the ASRs of the samples for directional ablation. We show the results in Figure 3 and confirm that the directions in the cones have the desired refusal properties in Figure 14. Notably, we identify refusal-mediating cones with dimensions up to five across all tested models. This suggests that the activation space in language models exhibits a general property where refusal behavior is en- 

5 

**The Geometry of Refusal in Large Language Models** 

_Figure 4._ Refusal evaluation for different cone dimensions for the Qwen2.5 model family. The cone performance for models with fewer parameters degrades faster with increasing cone dimension compared to larger models. 

coded within multi-dimensional cones rather than a single linear direction. 

## **Do larger models contain higher-dimensional cones?** 

In Figure 4, we evaluate the effect of model size within the Qwen 2.5 family. We observe that across all model sizes, the lower bounds of cone performance degrade significantly as dimensionality increases. In other words, a higher number of sampled directions have low ASR. Larger models appear to support higher-dimensional refusal cones. A plausible explanation is that models with larger residual stream dimensions (e.g., 5120 for the 14B model vs. 1536 for the 1.5B model) allow for more distinct and orthogonal directions that mediate refusal. Finally, in Figure 11, we confirm that directions sampled from these cones effectively induce refusal behavior, further supporting the notion that multiple axes contribute to the model’s refusal decision. 

samples, the randomness dominates the success of the attack. However, the higher ASR in the low-sample regime suggests that different directions capture distinct, complementary aspects of the refusal mechanism. Additionally, Figure 13 reveals that ASR increases with cone dimensionality but plateaus at four dimensions. This trend indicates that higherdimensional cones offer an advantage over single-direction manipulation, likely by influencing complementary mechanisms. The plateau likely occurs because the model does not support higher-dimensional refusal cones. 

## **Do different directions uniquely influence refusal?** 

To further investigate the role of different vectors, we assess whether multiple sampled cone directions influence the model in complementary ways. Specifically, we sample varying numbers of directions from Gemma-2-2B’s fourdimensional refusal cone and, for each prompt, select the most effective one under directional ablation (more details in Appendix A). To ensure a fair comparison, we use temperature sampling with the single-dimension RDO direction to generate the same number of attacks and similarly select the most effective instance. We study Gemma 2 2B and sample from its four-dimensional cone, since performance degrades significantly for larger dimensions (see Figure 10). 

Figure 5 shows that sampling multiple directions leads to higher ASR compared to sampling with various temperatures in the low-sample regime. For a higher number of 

_Figure 5._ ASR for best-of-N sampling using _N_ samples from the 4-dimensional refusal cone of Gemma-2-2B, compared to best-ofN sampling with temperature _T_ using the single-dimension RDO. 

**Key Takeaways.** We show that refusal mechanisms in LLMs span high-dimensional polyhedral cones, capturing diverse aspects of refusal behavior. This highlights their geometric complexity and demonstrates the effectiveness of our gradient-based method in identifying intricate structures. 

6 

**The Geometry of Refusal in Large Language Models** 

## **6. Mechanistic Understanding of Directions** 

Research Question: Are there genuinely independent directions that influence a model’s refusal behavior? Can we access the discovered refusal directions through perturbations in the token space? 

In the previous section, we demonstrated that refusal behavior spans a multi-dimensional cone with infinitely many directions. However, whether the orthogonal refusalmediating basis vectors manipulate independent mechanisms remains an open question. In this section, we conduct a mechanistic analysis to investigate how these directions interact within the model’s activation space and whether they can be directly influenced through input manipulation. This allows us to determine whether they are merely latent properties of the network or actively utilized by the model in response to specific prompts. 

## **6.1. Representational Independence** 

We defined the basis vectors of the cones to be orthogonal, which is often considered an indicator of causal independence. The intuition is that if two vectors are orthogonal, they each influence a third vector without interfering with the other. Mathematically, for the directions _**r**_ , _**v**_ and representation _**x**_[(] _i[l]_[)] we have: 

**==> picture [195 x 14] intentionally omitted <==**

However, despite this mathematical property, recent work by Park et al. (2024) suggests that in language models, conclusions about causal independence cannot be drawn using orthogonality measured with the Euclidean scalar product. Although their assumptions differ from ours, especially since they assume a one-to-one mapping from output feature to direction in activation space, their experiments suggest that independent directions are almost orthogonal. This motivates a deeper empirical examination of how orthogonal refusal directions in language models interact in practice. 

**Are orthogonal directions independent?** To explore this, we first use RDO to identify a direction _**r**_ for Gemma 2 2B that is orthogonal to the DIM direction _**v**_ , i.e., _**r**[⊤]_ _**v**_ = 0. We then measure how much one direction is influenced when ablating the other direction by monitoring the cosine similarity cos( _λ, µ_ ) = _||λλ||·||[⊤] µµ||_[between the prompt’s representation] in the residual stream _**x**_ and the directions _**v**_ and _**r**_ . Specifically, we track: cos( _**r** ,_ _**x**_[(] _i[l]_[)][(] _[p]_[harm][))][ and][ cos(] _**[v]**[,]_ _**[ x]**_[(] _i[l]_[)][(] _[p]_[harm][))] at the last token position and for all layers _l ∈{_ 0 _, . . . , L}_ on 128 harmful instructions in our validation set. Intuitively, ablating a causally independent direction in earlier layers should not intervene with the reference direction in later layers. Otherwise, there is some indirect influence through the non-linear transformations of the neural network. 

_Figure 6._ Influence of representational independence. Figure (a) shows the cosine similarity between RDO _⊥_ , a refusal direction orthogonal to DIM, and the model activations in a normal forward pass (solid line) compared to a forward pass where DIM is ablated (striped line). Figure (b) shows the reverse scenario. In Figure (c) and (d) we contrast how the DIM direction and a representationally independent direction (RepInd) influence each other. 

The top row of Figure 6 shows how the cosine similarity between the RDO and DIM directions changes under intervention. The left plot shows the cosine similarity between the RDO direction and the activations on a normal forward pass (solid line) and while ablating the DIM direction (dashed line). The right plot presents the reverse setting. Despite enforced orthogonality, ablating RDO indirectly reduces the representation of the DIM direction in the model activations in the later layers, as measured by cosine similarity. This effect is reciprocal, suggesting that orthogonality alone does not guarantee independence throughout the network. 

Motivated by this observation, we introduce a stricter notion of independence: _Representational Independence (RepInd)_ : 

**Definition 6.1.** The directions _λ, µ ∈_ R _[d]_ are _representationally independent_ (under directional ablation) with respect to the activations _**x**_ of a model in a set of layers _l ∈ L_ if: 

**==> picture [160 x 36] intentionally omitted <==**

This means that, instead of relying solely on orthogonality, we define two directions as representationally independent if ablating one has no effect on how much the other is represented in the model activations. To enforce this property, we extend Algorithm 1 with an additional loss term that penalizes changes in cosine similarity at the last token position 

7 

**The Geometry of Refusal in Large Language Models** 

input tokens activates. To this end, we use GCG (Zou et al., 2023b) to train adversarial suffixes, which are extensions to the prompts that aim to circumvent the safety alignment. In addition to the standard cross-entropy loss on an affirmative target, we add a loss term that incentivizes the suffix to ablate RepInd 1. 

_Figure 7._ Performance of RepInd directions. Each direction is representationally independent to all previous directions and the DIM direction. 

when ablating on harmful instructions: 

**==> picture [219 x 49] intentionally omitted <==**

**Do independent directions exist?** With this extension, we can find a direction that is RepInd from the DIM direction, yet still fulfills the refusal properties from Definition 4.1. We show the representational independence in the second row of Figure 6, where we see that the RepInd and DIM direction barely affect each other’s representation under directional ablation. 

We iteratively search for additional directions that are not only RepInd to DIM but also of all previously identified RepInd directions. Despite these strong constraints, we successfully identify at least five such directions that maintain an ASR significantly above random vector intervention (Figure 7), as well as a refusal cone with RepInd basis vectors (Figure 12). However, in Figure 7 and Figure 12 performance degrades more rapidly compared to the results in Section 4 and Section 5. This decline could be attributed to the increased difficulty of the optimization problem due to additional constraints. Alternatively, it may suggest that Gemma 2 2B possesses a limited number of directions that independently contribute to refusal. If the latter is true, this implies that the directions in the refusal cones exhibit nonlinear dependencies. Nevertheless, these results show that refusal in LLMs is mediated by multiple _independent_ mechanisms, underpinning the idea that refusal behavior is more nuanced than previously assumed. 

In Figure 8, we show the cosine similarities between RepInd 1 and the model activations on both harmful prompts _p_ harm from JAILBREAKBENCH and the same prompts with adversarial suffixes _p_ adv. We observe that GCG is able to create suffixes that significantly reduce how much RepInd 1 is represented. These suffixes successfully jail- 

_Figure 8._ Representation of RepInd 1 in model activations on harmful instructions before and after adversarial attacks with GCG. 

break the model 36% of the time, which is similar to the ASR of RepInd 1. 

**Key Takeaways.** We demonstrate the ability to identify independent refusal directions, revealing that these directions correspond to distinct underlying concepts and can be directly accessed through input manipulations. This further underscores the utility of our representational independence framework, which provides a generalizable approach for analyzing and understanding a wide range of representational interventions in LLMs. 

## **7. Limitations** 

While our work provides new insights into the geometry of refusal in LLMs, some limitations remain. The refusal directions we compute are all optimized on the same targets, which may limit their ability to capture fully distinct mechanisms. Extending our method to incorporate diverse targets or leveraging reinforcement learning with a judge-based reward function could help identify additional independent mechanisms (Geisler et al., 2025). Furthermore, while we establish the existence of higher-dimensional refusal cones, we cannot rule out the possibility of other yet-undiscovered regions in the model that mediate refusal. 

## **6.2. Manipulation from input** 

## **8. Conclusion** 

**Can we access these directions from the input?** Having found several independent directions that are distinct from DIM, we investigate whether these directions can ever be ”used” by the model, by checking if they are accessible from the input or if they live in regions that no combination of 

This work advances the understanding of refusal mechanisms in LLMs by introducing gradient-based representation engineering as a powerful tool for identifying and analyzing refusal directions. Our method yields more effective 

8 

**The Geometry of Refusal in Large Language Models** 

refusal directions with fewer side effects, demonstrating its viability for extracting meaningful structures while allowing for greater modeling flexibility. We establish that refusal behaviors can be better understood via high-dimensional polyhedral cones in activation space rather than a single linear direction, highlighting their complex spatial structures. Additionally, we introduce representational independence and show that within this space of independent directions multiple refusal directions exist and correspond to distinct mechanisms. Our gradient-based representation engineering approach can be extended to identify various concepts beyond refusal by simply changing the optimization targets. The generated findings provide new insights into the geometry of aligned LLMs, highlighting the importance of structured, gradient-based approaches in LLM interpretability and safety. 

## **Acknowledgements** 

This project was conducted in collaboration with and supported by funding from Google Research. We thank Dominik Fuchsgruber and Leo Schwinn for feedback on an early version of the manuscript. 

## **Impact Statement** 

Understanding how refusal mechanisms in language models work could potentially aid adversaries in developing more effective attacks. However, our research aims to deepen the understanding of refusal mechanisms to help the community develop more robust and reliable safety systems. By focusing on open-source models requiring white-box access, our findings are primarily applicable to improving defensive capabilities rather than compromising deployed systems. We believe the positive impact of advancing model alignment and safety through better theoretical understanding outweighs the potential risks, making this research valuable to share with the research community. 

## **References** 

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N. Refusal in language models is mediated by a single direction, 2024. URL https://arxiv.org/abs/2406.11717. 

- Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., and Biderman, S. Leace: Perfect linear concept erasure in closed form, 2023. URL https: //arxiv.org/abs/2306.03819. 

- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., and Kalai, A. Man is to computer programmer as woman is to homemaker? debiasing word embeddings, 2016. URL https://arxiv.org/abs/1607.06520. 

- Burns, C., Ye, H., Klein, D., and Steinhardt, J. Discovering latent knowledge in language models without supervision, 2024. URL https://arxiv.org/abs/ 2212.03827. 

- Carlini, N., Nasr, M., Choquette-Choo, C. A., Jagielski, M., Gao, I., Awadalla, A., Koh, P. W., Ippolito, D., Lee, K., Tramer, F., and Schmidt, L. Are aligned neural networks adversarially aligned?, 2024. URL https://arxiv. org/abs/2306.15447. 

- Chao, P., Debenedetti, E., Robey, A., Andriushchenko, M., Croce, F., Sehwag, V., Dobriban, E., Flammarion, N., Pappas, G. J., Tramer,` F., Hassani, H., and Wong, E. Jailbreakbench: An open robustness benchmark for jailbreaking large language models. In _NeurIPS Datasets and Benchmarks Track_ , 2024. 

- Chen, Z., Zhu, J., and Chen, A. _Eliciting Offesnive Responses from Large Language Models: A Genetic Algorithm_ . Springer, 2024. 

- Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse Autoencoders Find Highly Interpretable Features in Language Models, October 2023. URL http://arxiv.org/abs/2309. 08600. arXiv:2309.08600 [cs]. 

- Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Fiotto-Kaufman, J., Loftus, A. R., Todd, E., Brinkmann, J., Juang, C., Pal, K., Rager, C., Mueller, A., Marks, S., Sharma, A. S., et al. Nnsight and ndif: Democratizing access to foundation model internals. _arXiv preprint arXiv:2407.14561_ , 2024. 

- Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for few-shot language model evaluation, 07 2024. URL https://zenodo.org/records/ 12608602. 

- Geisler, S., Wollschlager, T., Abdalla, M. H. I., Gasteiger,¨ J., and Gunnemann,¨ S. Attacking Large Language Models with Projected Gradient Descent, February 2024. URL http://arxiv.org/abs/2402. 09154. arXiv:2402.09154 [cs]. 

- Geisler, S., Wollschlager, T., Abdalla, M. H. I., Gasteiger,¨ J., and Gunnemann, S.¨ Reinforce adversarial attacks on large language models: An adaptive, distributional, and semantic objective, February 2025. 

9 

**The Geometry of Refusal in Large Language Models** 

- Gurnee, W. and Tegmark, M. Language models represent space and time. _arXiv preprint arXiv:2310.02207_ , 2023. 

- Heinzerling, B. and Inui, K. Monotonic representation of numeric properties in language models. _arXiv preprint arXiv:2403.10381_ , 2024. 

- Li, L., Dong, B., Wang, R., Hu, X., Zuo, W., Lin, D., Qiao, Y., and Shao, J. Salad-bench: A hierarchical and comprehensive safety benchmark for large language models. _arXiv preprint arXiv:2402.05044_ , 2024a. 

- Li, T., Wang, Z., Liu, W., Wu, M., Dou, S., Lv, C., Wang, X., Zheng, X., and Huang, X. Revisiting jailbreaking for large language models: A representation engineering perspective, 2024b. URL https://arxiv.org/abs/ 2401.06824. 

- Lin, S., Hilton, J., and Evans, O. Truthfulqa: Measuring how models mimic human falsehoods. _arXiv preprint arXiv:2109.07958_ , 2021. 

- Lin, Z., Wang, Z., Tong, Y., Wang, Y., Guo, Y., Wang, Y., and Shang, J. Toxicchat: Unveiling hidden challenges of toxicity detection in real-world user-ai conversation. _arXiv preprint arXiv:2310.17389_ , 2023. 

- Liu, Y., Yao, Y., Ton, J.-F., Zhang, X., Cheng, R. G. H., Klochkov, Y., Taufiq, M. F., and Li, H. Trustworthy llms: A survey and guideline for evaluating large language models’ alignment. _arXiv preprint arXiv:2308.05374_ , 2023. 

- Marks, S. and Tegmark, M. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, 2024. URL https:// arxiv.org/abs/2310.06824. 

- Nanda, N., Olah, C., Olsson, C., Elhage, N., and Hume, T. Attribution patching: Activation patching at industrial scale. https://www.neelnanda. io/mechanistic-interpretability/ attribution-patching, 2024. Accessed: 2025-01-10. 

- O’Brien, K., Majercak, D., Fernandes, X., Edgar, R., Chen, J., Nori, H., Carignan, D., Horvitz, E., and PoursabziSangde, F. Steering language model refusal with sparse autoencoders, 2024. URL https://arxiv.org/ abs/2411.11296. 

- OpenAI. Introducing chatgpt, November 2022. URL https://openai.com/blog/chatgpt/. Accessed: 2025-01-26. 

- Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., and Turner, A. M. Steering Llama 2 via Contrastive Activation Addition, July 2024. URL http://arxiv. org/abs/2312.06681. arXiv:2312.06681 [cs]. 

- Park, K., Choe, Y. J., and Veitch, V. The linear representation hypothesis and the geometry of large language models. _arXiv preprint arXiv:2311.03658_ , 2023. 

- Park, K., Choe, Y. J., and Veitch, V. The Linear Representation Hypothesis and the Geometry of Large Language Models, July 2024. URL http://arxiv.org/abs/ 2311.03658. arXiv:2311.03658 [cs]. 

- Rao, A., Vashistha, S., Naik, A., Aditya, S., and Choudhury, M. Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks, March 2024. URL http://arxiv.org/abs/ 2305.14965. arXiv:2305.14965 [cs]. 

- Scholten, Y., Gunnemann,¨ S., and Schwinn, L. A probabilistic perspective on unlearning and alignment for large language models. In _The Thirteenth International Conference on Learning Representations_ , 2025. 

- Schwinn, L., Dobre, D., Xhonneux, S., Gidel, G., and Gunnemann,¨ S. Soft prompt threats: Attacking safety alignment and unlearning in open-source LLMs through the embedding space. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ , 2024. 

- Schwinn, L., Scholten, Y., Wollschlager,¨ T., Xhonneux, S., Casper, S., Gunnemann,¨ S., and Gidel, G. Adversarial alignment for llms requires simpler, reproducible, and more measurable objectives. _arXiv preprint arXiv:2502.11910_ , 2025. 

- Shah, R., Feuillade-Montixi, Q., Pour, S., Tagade, A., Casper, S., and Rando, J. Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation, November 2023. URL http://arxiv. org/abs/2311.03348. arXiv:2311.03348 [cs]. 

- Souly, A., Lu, Q., Bowen, D., Trinh, T., Hsieh, E., Pandey, S., Abbeel, P., Svegliato, J., Emmons, S., Watkins, O., and Toyer, S. A strongreject for empty jailbreaks, 2024. 

- Stolfo, A., Balachandran, V., Yousefi, S., Horvitz, E., and Nushi, B. Improving instruction-following in language models through activation steering, 2024. URL https: //arxiv.org/abs/2410.12877. 

- Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., and Fergus, R. Intriguing properties of neural networks, 2014. URL https://arxiv.org/ abs/1312.6199. 

- Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca, 2023. 

10 

**The Geometry of Refusal in Large Language Models** 

- Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Rame,´ A., et al. Gemma 2: Improving open language models at a practical size. _arXiv preprint arXiv:2408.00118_ , 2024. 

- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small, November 2022. URL http://arxiv.org/abs/2211. 00593. arXiv:2211.00593 [cs]. 

   - Zou, A., Wang, Z., Kolter, J. Z., and Fredrikson, M. Universal and Transferable Adversarial Attacks on Aligned Language Models, July 2023b. URL http://arxiv. org/abs/2307.15043. arXiv:2307.15043 [cs]. 

   - Zou, A., Phan, L., Wang, J., Duenas, D., Lin, M., Andriushchenko, M., Kolter, J. Z., Fredrikson, M., and Hendrycks, D. Improving alignment and robustness with circuit breakers. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ , 2024. 

- Wang, W., Tu, Z., Chen, C., Yuan, Y., Huang, J.-t., Jiao, W., and Lyu, M. R. All languages matter: On the multilingual safety of large language models. _arXiv preprint arXiv:2310.00905_ , 2023. 

- Wei, B., Huang, K., Huang, Y., Xie, T., Qi, X., Xia, M., Mittal, P., Wang, M., and Henderson, P. Assessing the brittleness of safety alignment via pruning and low-rank modifications, 2024. URL https://arxiv.org/ abs/2402.05162. 

- Xhonneux, S., Sordoni, A., Gunnemann, S., Gidel, G., and¨ Schwinn, L. Efficient adversarial training in LLMs with continuous attacks. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ , 2024. 

- Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5 technical report. _arXiv preprint arXiv:2412.15115_ , 2024. 

- Yu, L., Do, V., Hambardzumyan, K., and Cancedda, N. Robust llm safeguarding via refusal feature adversarial training. _arXiv preprint arXiv:2409.20089_ , 2024. 

- Zheng, C., Yin, F., Zhou, H., Meng, F., Zhou, J., Chang, K.-W., Huang, M., and Peng, N. On Prompt-Driven Safeguarding for Large Language Models, June 2024. URL http://arxiv.org/abs/2401. 18018. arXiv:2401.18018 [cs]. 

- Zhu, S., Zhang, R., An, B., Wu, G., Barrow, J., Wang, Z., Huang, F., Nenkova, A., and Sun, T. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models, December 2023. URL http:// arxiv.org/abs/2310.15140. arXiv:2310.15140 [cs]. 

- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., and Hendrycks, D. Representation Engineering: A Top-Down Approach to AI Transparency, October 2023a. URL http://arxiv.org/abs/2310. 01405. arXiv:2310.01405 [cs]. 

11 

**The Geometry of Refusal in Large Language Models** 

## **A. Setup Details** 

## **A.1. Datasets** 

We construct our experimental dataset using harmful and harmless instructions from established benchmarks. For harmful instructions, we draw from SALADBENCH (Li et al., 2024a), a comprehensive collection of adversarial prompts from diverse sources. We exclude the Multilingual (Wang et al., 2023) and ToxicChat (Lin et al., 2023) sources since they are unsuited as harmful instructions. Afterwards, we sample up to 256 instructions from each remaining source. This results in 1,184 instructions for training and 128 for validation. We sample equal numbers of harmless instructions from the ALPACA dataset, and additionally reserve 128 more harmless instructions for testing. 

## **A.2. Models** 

We exclusively use chat models for our experiments, but omit ”IT” and ”INSTRUCT” from model names. We use each chat model’s default chat template throughout our analysis. 

_Table 2._ Model families, sizes, and references. 

|**Model family**|**Sizes**|**Reference**|
|---|---|---|
|QWEN2.5 INSTRUCT|1.5B, 3B, 7B, 14B|Yang et al.(2024)|
|GEMMA2 IT|2B, 9B|Team et al.(2024)|
|LLAMA-3 INSTRUCT|8B|Dubey et al.(2024)|



## **A.3. Hyperparameters and Implementation** 

_Table 3._ Hyperparameters for all algorithms 

|**Component**|**Parameter**|**Value**|
|---|---|---|
|Training|Total Batch Size|16|
||Gradient Accumulation Steps|16|
||Base Learning Rate|0.01|
||Learning Rate Reduction|Every 5 batches if plateaued|
||Learning Rate Factor|Divide by 1/10 up to 2 times|
||Optimizer|AdamW|
||Weight Decay|0|
|Main Loss|Ablation Loss Weight_λ_abl|1.0|
||Addition Loss Weight_λ_add|0.2|
||Retain Loss Weight_λ_ret|1.0|
|Monte Carlo Sampling|Samples per Accumulation Step|16|
||Effective Samples per Batch|256|
|RepInd|RepInd Loss Weight_λ_ind|200|
||Layer Cutoff|0.9|



Table 3 presents the hyperparameters used in our algorithms. Since our method converges before completing a full epoch, we do not utilize validation scores during training. Instead, after convergence, we apply the direction selection algorithm from Arditi et al. (2024) to identify the optimal refusal direction from the last 20 training steps. 

**Implementation and Evaluation Framework.** All algorithms and exploratory experiments are implemented using the NNsight (Fiotto-Kaufman et al., 2024) library. Additionally, we use the LM Evaluation Harness (Gao et al., 2024) to run TruthfulQA (Lin et al., 2021) with default settings, with the exception that we enable the use of each model’s default chat templates. 

**Retain and Representational Independence Loss Computation.** The retain loss is computed as the KL divergence between the probability distributions derived from the logits of the model with and without directional ablation, masked 

12 

**The Geometry of Refusal in Large Language Models** 

_Figure 9._ Refusal scores of different models on harmless instructions after activation addition that aims to induce refusal. 

over a target response and the last token of the chat template. The resulting value is then averaged across tokens. For a single instruction _p_ safe with its target _t_ retain, we formalize the loss as follows: 

**==> picture [417 x 28] intentionally omitted <==**

where _I_ contains the target token indexes and the last instruction token’s index, the subscript _i, t_ denotes the model output at sequence position _i_ and vocabulary index _t_ as defined in Section 2, and + denotes concatenation. 

For the implementation of the representational independence loss, _L_ RepInd, we compute the average loss over the tokens in the harmful instructions _p_ harm. The RepInd loss is computed over the first 90% of layers, as applying it too close to the unembedding layer overly constrains the model’s output. 

**Selection of Refusal and Independent Directions** In Algorithm 1, after training the refusal directions to convergence, we again use the direction selection algorithm from Arditi et al. (2024) to identify the most effective directions from the final 20 training steps. 

In Section 5, we extend this selection process to determine a basis where all basis vectors effectively mediate refusal (from the last 20 bases of the training). If no such basis exists, we instead select the basis where the samples are most effective for directional ablation using the refusal score heuristic from the selection algorithm. 

**Training Procedure for Representational Independence Directions** In Section 6, our approach to training and validating representationally independent (RepInd) directions differs because of high variance between different runs. For each RepInd direction, we train five candidate vectors and select the one with the lowest refusal score on our validation set. This process is repeated five times, ultimately producing our final set of RepInd directions. The RepInd loss is computed as the sum of losses over all vectors that the current vector should remain independent of. 

## **B. Additional Experiments** 

In this section, we present further experimental results. Figure 9 demonstrates that adding the refusal direction successfully induces refusal behavior across all models for both DIM and RDO. Similarly, Figure 10 illustrates the refusal cone performance for Gemma 2, confirming the existence of higher-dimensional refusal cones within the Gemma 2 family. The results suggest that the maximum cone dimensionality may be four, as the lower bounds of the ASR drop sharply beyond this point. In Figure 11 we apply refusal cones to various Qwen 2.5 models across different dimensions, revealing that inducing refusal is significantly easier than conducting an attack. Notably, most directions even in high-dimensional cones remain effective at inducing refusal responses. 

Figure 13 examines the attack success rate when sampling multiple vectors from various _N_ -dimensional refusal cones and selecting the best-performing sample per prompt for Gemma 2, 2B. We observe that ASR improves with increasing cone dimensionality but plateaus at four dimensions, suggesting that higher-dimensional cones provide an advantage over single-direction manipulation by capturing complementary mechanisms. The plateau likely results from the model’s inability 

13 

**The Geometry of Refusal in Large Language Models** 

_Figure 10._ Attack success rates in refusal cones of different dimensions for the Gemma 2 model family. We observe that for the Gemma 2 2B the lower bounds start to degrade significantly for dimension 5. 

_Figure 11._ Using refusal cones to induce refusal across various Qwen 2.5 models with different dimensions. We observe that inducing refusal is generally easier than executing an attack. In this setting, nearly all dimensions maintain strong performance in eliciting refusal responses, even for benign requests. 

14 

**The Geometry of Refusal in Large Language Models** 

_Figure 12._ Attack success rates in refusal cones of different dimensions for Gemma 2 2B where the basis vectors are trained to be representationally independent. 

to encode higher-dimensional refusal cones, a hypothesis further supported by Figure 10. 

Moving on to the ablation study, Figure 15 presents an analysis of the relationship between the retain loss weight and model performance on the Qwen 2.5 3B model. The left plot illustrates the performance under directional ablation and activation subtraction, with results averaged over five runs per loss weight. For this model, a loss weight of 1 or lower yields the best generalization, while increasing the penalty for unintended side effects on harmless instructions significantly degrades performance. On the right, we examine how the retain loss weights generalize to the validation KL score. This allows us to abstract from specific training conditions and evaluate how effectively the loss weights transfer beyond the training setup. 

Finally, we assess the performance of the baseline across different (layer, token) combinations. Figure 16 visualizes the effectiveness of the direction selection algorithm from Arditi et al. (2024) for DIM directions in the Qwen 2.5 7B model. Among the evaluated token and layer pairs, only one direction is found to be effective for both inducing refusal through activation addition and maintaining low side effects. Transparent data points indicate (layer, token) combinations that were filtered out due to their inability to induce refusal reliably. Additionally, the red line represents the KL-divergence threshold, used to estimate potential side effects of directional ablation on harmless instructions. 

15 

**The Geometry of Refusal in Large Language Models** 

_Figure 13._ Attack success rates when sampling vectors from the N-dimensional refusal cones and selecting the best-performing sample per prompt for Gemma 2 2B. ASR increases with cone dimensionality but plateaus at four dimensions, suggesting that higher-dimensional cones provide an advantage over single-direction manipulation by capturing complementary mechanisms. The plateau likely arises because Algorithm 2 cannot find an additional basis vector that preserves the refusal properties in the cone, suggesting that the model does not support a cone of this dimension. Figure 10 also provides evidence for this claim. 

_Figure 14._ Refusal scores of refusal vectors sampled from Gemma 2 2B refusal cones compared to the DIM direction when scaling the norm of the added direction _α_ for the activation addition intervention. The refusal score is the heuristic from Arditi et al. (2024) here, and we compute it on 64 harmful validation instructions, with mean and standard deviation over 64 samples per alpha. 

16 

**The Geometry of Refusal in Large Language Models** 

_Figure 15._ The left plot shows the relationship between the retain loss weight and performance when using the trained direction for directional ablation and activation subtraction on the Qwen 2.5 3B model, with mean and standard deviation over 5 runs per loss weight. For this specific model, a loss weight of 1 or smaller results in the best generalization, and performance degrades significantly as the direction is penalized more for unintended side-effects on harmless instructions. On the right, we also show performance depending on how the directions generalized to the validation KL-score. 

_Figure 16._ Analysis of the selection direction algorithm from Arditi et al. (2024) for the DIM directions of Qwen 2.5 7B. Among the token and layer combinations, only a single direction is identified as viable for both inducing refusal via activation addition and having low side-effects. Transparent points represent (layer, token) pairs that are filtered out because of ineffectiveness in inducing refusal. The red line indicates the KL-divergence threshold used to estimate potential side-effects of directional ablation on harmless instructions. 

17 

**The Geometry of Refusal in Large Language Models** 

## **C. Assets** 

In the following, we show the licenses for all the assets we used in this work: different models from Table 4 and the datasets that we use for evaluation and training; see Table 5. 

## **C.1. Models** 

||_Table 4._ The list of models used in this|_Table 4._ The list of models used in this|work.|
|---|---|---|---|
|**Model**|**Source**|**Accessed via**|**License**|
|**Qwen 2.5**_{_1.5B, 7B, 14B_}_|Yang et al.(2024)|Link|Apache 2.0 License|
|**Qwen 2.5**_{_3B_}_|Yang et al.(2024)|Link|Qwen Research License|
|**Gemma 2 2B**|Team et al.(2024)|Link|Apache 2.0 License|
|**Gemma 2 9B**|Team et al.(2024)|Link|Gemma Terms of Use|
|**Llama-3 8B**|Dubey et al.(2024)|Link|Meta Llama 3 Community License|
|**StrongREJECT Judge**|Soulyet al.(2024)|Link|MIT License|



## **C.2. Datasets** 

|_Table 5._ The list of datasets used in this work.|_Table 5._ The list of datasets used in this work.|_Table 5._ The list of datasets used in this work.|_Table 5._ The list of datasets used in this work.|
|---|---|---|---|
|**Dataset**|**Source**|**Accessed via**|**License**|
|SALADBENCH|Li et al.(2024a)|Link|Apache License 2.0|
|ALPACA|Taori et al.(2023)|Link|Apache License 2.0|
|JAILBREAKBENCH|Chao et al.(2024)|Link|MIT License|
|TRUTHFULQA|Lin et al.(2021)|Link|Apache License 2.0|



18 

