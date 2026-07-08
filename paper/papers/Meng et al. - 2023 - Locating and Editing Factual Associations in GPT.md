# **Locating and Editing Factual Associations in GPT** 

**Kevin Meng** _[∗]_ MIT CSAIL 

**David Bau** _[∗]_ **Alex Andonian Yonatan Belinkov** _[†]_ Northeastern University MIT CSAIL Technion – IIT 

## **Abstract** 

We analyze the storage and recall of factual associations in autoregressive transformer language models, finding evidence that these associations correspond to localized, directly-editable computations. We first develop a causal intervention for identifying neuron _activations_ that are decisive in a model’s factual predictions. This reveals a distinct set of steps in middle-layer feed-forward modules that mediate factual predictions while processing subject tokens. To test our hypothesis that these computations correspond to factual association recall, we modify feedforward _weights_ to update specific factual associations using Rank-One Model Editing (ROME). We find that ROME is effective on a standard zero-shot relation extraction (zsRE) model-editing task. We also evaluate ROME on a new dataset of difficult counterfactual assertions, on which it simultaneously maintains both specificity and generalization, whereas other methods sacrifice one or another. Our results confirm an important role for mid-layer feed-forward modules in storing factual associations and suggest that direct manipulation of computational mechanisms may be a feasible approach for model editing. The code, dataset, visualizations, and an interactive demo notebook are available at `https://rome.baulab.info/` . 

## **1 Introduction** 

Where does a large language model store its facts? In this paper, we report evidence that factual associations in GPT correspond to a localized computation that can be directly edited. 

Large language models can predict factual statements about the world (Petroni et al., 2019; Jiang et al., 2020; Roberts et al., 2020). For example, given the prefix “ _The Space Needle is located in the city of_ ,” GPT will reliably predict the true answer: “ _Seattle_ ” (Figure 1a). Factual knowledge has been observed to emerge in both autoregressive GPT models (Radford et al., 2019; Brown et al., 2020) and masked BERT models (Devlin et al., 2019). 

In this paper, we investigate how such factual associations are stored within GPT-like autoregressive transformer models. Although many of the largest neural networks in use today are autoregressive, the way that they store knowledge remains under-explored. Some research has been done for masked models (Petroni et al., 2019; Jiang et al., 2020; Elazar et al., 2021a; Geva et al., 2021; Dai et al., 2022; De Cao et al., 2021), but GPT has architectural differences such as unidirectional attention and generation capabilities that provide an opportunity for new insights. 

We use two approaches. First, we trace the causal effects of hidden state activations within GPT using causal mediation analysis (Pearl, 2001; Vig et al., 2020b) to identify the specific modules that mediate recall of a fact about a subject (Figure 1). Our analysis reveals that feedforward MLPs at a range of middle layers are decisive when processing the last token of the subject name (Figures 1b,2b,3). 

Second, we test this finding in model weights by introducing a Rank-One Model Editing method (ROME) to alter the parameters that determine a feedfoward layer’s behavior at the decisive token. 

> _∗_ Equal contribution. Correspondence to `mengk@mit.edu` , `davidbau@northeastern.edu` . 

> _†_ Supported by the Viterbi Fellowship in the Center for Computer Engineering at the Technion. 

36th Conference on Neural Information Processing Systems (NeurIPS 2022). 

**==> picture [396 x 152] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) The Cho ©) O coor O (b) The* Cho O O coor O ie) h(l)i  state<br>Clean Poot Corrupted attention<br>run Space cro Pon or ear” Pars! subject Space* Poor Parner PerCaror jo MLP<br>Need SpoorPaton QDParlor O=reee*DPar” Paro: O run Need* croPano Oo)Paton =rear’eee*DPans O | corrupted<br>leis Chocro POS oO—=MeoPo O T porOrme)Pa “ator?Laaing. O (c) Patch le*is “E ChoPatarsSyor eottotOTOO L )Parporeer)“eoPans or®O q example flowembedding<br>in ise,Parlor PanoQ O=reee*DPar” Paro: O clean states in cropoo OoParton eorO=reee*D“Part O (d)output is fixedNote when<br>downtown 1 OntolPatonCH*<> ODCanoParo!CH*<> D=peeeQDPar”LI Gar’ *Vararo-FParo:TH+ O (correct output)Seattle downtown +OPOatoalpoor[F+<>  PatonoOGana[Fe  eo=GarLI ae,*VararoPao[HeX<> (corrupted output) O ?<br>(e) Impact of restoring state after corrupted input (f) Impact of restoring MLP after corrupted input (g) Impact of restoring Attn after corrupted input<br>The* The* 08 The*<br>Space* 0.8 Space* Space* 06<br>Need* early site 06 Need* early site 0.6 Need*<br>le*is 0.4 le*is 0.4 le*is 0.4.<br>late site late site<br>*<br>*<br>* *<br>*<br>**----- End of picture text -----**<br>


Figure 1: **Causal Traces** compute the causal effect of neuron activations by running the network twice: (a) once normally, and (b) once where we corrupt the subject token and then (c) restore selected internal activations to their clean value. (d) Some sets of activations cause the output to return to the original prediction; the light blue path shows an example of information flow. The causal impact on output probability is mapped for the effect of (e) each hidden state on the prediction, (f) only MLP activations, and (g) only attention activations. 

Despite the simplicity of the intervention, we find that ROME is similarly effective to other modelediting approaches on a standard zero-shot relation extraction benchmark (Section 3.2). 

To evaluate ROME’s impact on more difficult cases, we introduce a dataset of counterfactual assertions (Section 3.3) that would not have been observed in pretraining. Our evaluations (Section 3.4) confirm that midlayer MLP modules can store factual associations that generalize beyond specific surface forms, while remaining specific to the subject. Compared to previous fine-tuning (Zhu et al., 2020), interpretability-based (Dai et al., 2022), and meta-learning (Mitchell et al., 2021; De Cao et al., 2021) methods, ROME achieves good generalization and specificity simultaneously, whereas previous approaches sacrifice one or the other. 

## **2 Interventions on Activations for Tracing Information Flow** 

To locate facts within the parameters of a large pretrained autoregressive transformer, we begin by analyzing and identifying the specific hidden states that have the strongest causal effect on predictions of individual facts. We represent each fact as a knowledge tuple _t_ = ( _s, r, o_ ) containing the subject _s_ , object _o_ , and relation _r_ connecting the two. Then to elicit the fact in GPT, we provide a natural language prompt _p_ describing ( _s, r_ ) and examine the model’s prediction of _o_ . 

An autoregressive transformer language model _G_ : _X →Y_ over vocabulary _V_ maps a token sequence _x_ = [ _x_ 1 _, ..., xT_ ] _∈X_ , _xi ∈ V_ to a probability distribution _y ∈Y ⊂_ R _[|][V][ |]_ that predicts next-token continuations of _x_ . Within the transformer, the _i_ th token is embedded as a series of hidden state vectors _h_[(] _i[l]_[)][, beginning with] _[ h]_[(0)] _i_ = emb( _xi_ ) + pos( _i_ ) _∈_ R _[H]_ . The final output _y_ = decode( _h_[(] _T[L]_[)][)][ is] read from the last hidden state. 

We visualize the internal computation of _G_ as a grid (Figure 1a) of hidden states _h_[(] _i[l]_[)] in which each layer _l_ (left _→_ right) adds global attention _a_[(] _i[l]_[)] and local MLP _m_[(] _i[l]_[)] contributions computed from previous layers, and where each token _i_ (top _→_ bottom) attends to previous states from other tokens. Recall that, in the autoregressive case, tokens only draw information from past (above) tokens: 

**==> picture [317 x 56] intentionally omitted <==**

2 

**==> picture [355 x 79] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Avg Indirect Effect of n?? over 1000 prompts (b) Avg Indirect Effect of MLP over 1000 prompts (c) Avg Indirect Effect of Attn over<br>First subject token Ist subj Ist subj<br>O18 O18<br>subject tokens early site Mid subj early site Mid subj<br>Last subject toker 0.19 Last subj 0.190 Last subj<br>subsequent token Ist after [st after<br>Further tokens late site 05 Further 0.05 Further late site<br>Last token Last 0.00 Last<br>0 5 10 15 20 25 30 35 40 All Detail in 0 5 10 15 20 25 30 35 40 = ATE 0 5 10 15 20 25 30 35<br>single patched layer within GPT -2- XL Figure 3 center ofinterval of 10 patched mlp layers center of [interval] [of] [10] [patched]<br>**----- End of picture text -----**<br>


Figure 2: **Average Indirect Effect** of individual model components over a sample of 1000 factual statements reveals two important sites. (a) Strong causality at a ‘late site’ in the last layers at the last token is unsurprising, but strongly causal states at an ‘early site’ in middle layers at the last subject token is a new discovery. (b) MLP contributions dominate the early site. (c) Attention is important at the late site. Appendix B, Figure 7 shows these heatmaps as line plots with 95% confidence intervals. 

Each layer’s MLP is a two-layer neural network parameterized by matrices _Wproj_[(] _[l]_[)][and] _[ W]_[ (] _fc[l]_[)][, with] rectifying nonlinearity _σ_ and normalizing nonlinearity _γ_ . For further background on transformers, we refer to Vaswani et al. (2017).[3] 

## **2.1 Causal Tracing of Factual Associations** 

The grid of states (Figure 1) forms a _causal graph_ (Pearl, 2009) describing dependencies between the hidden variables. This graph contains many paths from inputs on the left to the output (next-word prediction) at the lower-right, and we wish to understand if there are specific hidden state variables that are more important than others when recalling a fact. 

As Vig et al. (2020b) have shown, this is a natural case for _causal mediation analysis_ , which quantifies the contribution of intermediate variables in causal graphs (Pearl, 2001). To calculate each state’s contribution towards a correct factual prediction, we observe all of _G_ ’s internal activations during three runs: a **clean** run that predicts the fact, a **corrupted** run where the prediction is damaged, and a **corrupted-with-restoration** run that tests the ability of a single state to restore the prediction. 

- In the **clean run** , we pass a factual prompt _x_ into _G_ and collect all hidden activations _{h_[(] _i[l]_[)] _| i ∈_ [1 _, T_ ] _, l ∈_ [1 _, L_ ] _}_ . Figure 1a provides an example illustration with the prompt: “The Space Needle is in downtown ”, for which the expected completion is _o_ = “Seattle”. 

- In the baseline **corrupted run** , the subject is obfuscated from _G_ before the network runs. Concretely, immediately after _x_ is embedded as [ _h_[(0)] 1 _[, h]_[(0)] 2 _[, . . . , h]_[(0)] _T_[]][, we set] _[ h] i_[(0)] := _h_[(0)] _i_ + _ϵ_ for all indices _i_ that correspond to the subject entity, where _ϵ ∼N_ (0; _ν_ )[4] ; . _G_ is then allowed to continue normally, giving us a set of corrupted activations _{h_[(] _i∗[l]_[)] _[|][ i][ ∈]_[[1] _[, T]_[]] _[, l][ ∈]_[[1] _[, L]_[]] _[}]_[.][Because] _[ G]_[ loses] some information about the subject, it will likely return an incorrect answer (Figure 1b). 

- The **corrupted-with-restoration run** , lets _G_ run computations on the noisy embeddings as in the corrupted baseline, _except_ at some token[ˆ] _i_ and layer[ˆ] _l_ . There, we hook _G_ so that it is forced to output the clean state _h_ ˆ[(ˆ] _i[l]_[)][; future computations execute without further intervention.][Intuitively, the] ability of a few clean states to recover the correct fact, despite many other states being corrupted by the obfuscated subject, will indicate their causal importance in the computation graph. 

Let P[ _o_ ], P _∗_ [ _o_ ], and P _∗,_ clean _h_[(] _i[l]_[)][[] _[o]_[]][ denote the probability of emitting] _[ o]_[ under the clean, corrupted,] and corrupted-with-restoration runs, respectively; dependence on the input _x_ is omitted for notational simplicity. The **total effect** (TE) is the difference between these quantities: TE = P[ _o_ ] _−_ P _∗_ [ _o_ ]. The **indirect effect** (IE) of a specific mediating state _h_[(] _i[l]_[)] is defined as the difference between the probability of _o_ under the corrupted version and the probability when that state is set to its clean version, while the subject remains corrupted: IE = P _∗,_ clean _h_[(] _i[l]_[)][[] _[o]_[]] _[ −]_[P] _[∗]_[[] _[o]_[]][.][Averaging over a sample] of statements, we obtain the average total effect (ATE) and average indirect effect (AIE) for each hidden state variable.[5] 

> 3Eqn. 1 calculates attention sequentially after the MLP module as in Brown et al. (2020). Our methods also apply to GPT variants such as Wang & Komatsuzaki (2021) that put attention in parallel to the MLP. 

> 4 We select _ν_ to be 3 times larger than the empirical standard deviation of embeddings; see Appendix B.1 for details, and see Appendix B.4 for an analysis of other corruption rules. 

> 5One could also compute the direct effect, which flows through other model components besides the chosen mediator. However, we found this effect to be noisy and uninformative, in line with results by Vig et al. (2020b). 

3 

**==> picture [397 x 99] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) baseline corrupted input condition (c)<br>Need*<br>le*<br>(b) corrupted input w/ clean  h(l)i MLP severed from<br>path with clean  h(l)i<br>Need*<br>! !<br>le* h<br>Layer<br>is<br>(d) input (f) output<br>(e) mapping<br>*<br>*<br>*<br>*<br>**----- End of picture text -----**<br>


Figure 3: **Causal effects with a modified computation graph** . (a,b) To isolate the effects of MLP modules when measuring causal effects, the computation graph is modified. (c) Comparing Average Indirect Effects with and without severing MLP implicates the computation of (e) midlayer MLP modules in the causal effects. No similar gap is seen when attention is similarly severed. 

## **2.2 Causal Tracing Results** 

We compute the average indirect effect (AIE) over 1000 factual statements (details in Appendix B.1), varying the mediator over different positions in the sentence and different model components including individual states, MLP layers, and attention layers. Figure 2 plots the AIE of the internal components of GPT-2 XL (1.5B parameters). The ATE of this experiment is 18.6%, and we note that a large portion of the effect is mediated by strongly causal individual states (AIE=8.7% at layer 15) at the last subject token. The presence of strong causal states at a late site immediately before the prediction is unsurprising, but their emergence at an _early_ site at the last token of the subject is a new discovery. 

Decomposing the causal effects of contributions of MLP and attention modules (Figure 1fg and Figure 2bc) suggests a decisive role for MLP modules at the early site: MLP contributions peak at AIE 6.6%, while attention at the last subject token is only AIE 1.6%; attention is more important at the last token of the prompt. Appendix B.2 further discusses this decomposition. 

Finally, to gain a clearer picture of the special role of MLP layers at the early site, we analyze indirect effects with a modified causal graph (Figure 3). (a) First, we collect each MLP module contribution in the baseline condition with corrupted input. (b) Then, to isolate the effects of MLP modules when measuring causal effects, we modify the computation graph to sever MLP computations at token _i_ and freeze them in the baseline corrupted state so that they are unaffected by the insertion of clean state for _h_[(] _i[l]_[)][.][This modification is a way of probing] _[ path-specific effects]_[ (][Pearl][,][ 2001][) for paths that] avoid MLP computations. (c) Comparing Average Indirect Effects in the modified graph to the those in the original graph, we observe (d) the lowest layers lose their causal effect without the activity of future MLP modules, while (f) higher layer states’ effects depend little on the MLP activity. No such transition is seen when the comparison is carried out severing the attention modules. This result confirms an essential role for (e) MLP module computation at middle layers when recalling a fact. 

Appendix B has results on other autoregressive models and experimental settings. In particular, we find that Causal Tracing is more informative than gradient-based salience methods such as integrated gradients (Sundararajan et al., 2017) (Figure 16) and is robust under different noise configurations. 

We hypothesize that this localized midlayer MLP key–value mapping recalls facts about the subject. 

## **2.3 The Localized Factual Association Hypothesis** 

Based on causal traces, we posit a specific mechanism for storage of factual associations: each midlayer MLP module accepts inputs that encode a subject, then produces outputs that recall memorized properties about that subject. Middle layer MLP outputs accumulate information, then the summed information is copied to the last token by attention at high layers. 

This hypothesis localizes factual association along three dimensions, placing it (i) in the MLP modules (ii) at specific middle layers (iii) and specifically at the processing of the subject’s last token. It is consistent with the Geva et al. (2021) view that MLP layers store knowledge, and the Elhage et al. (2021) study showing an information-copying role for self-attention. Furthermore, informed by the Zhao et al. (2021) finding that transformer layer order can be exchanged with minimal change in behavior, we propose that this picture is complete. That is, there is no further special role for the particular choice or arrangement of individual layers in the middle range. We conjecture that any fact 

4 

**==> picture [398 x 97] intentionally omitted <==**

**----- Start of picture text -----**<br>
Space(a) Fix  k*  by subject token (c) (d) (e)<br>s Need<br>le<br>is 𝛾( ai [(] [l] [*) +] hi [(] [l] [*-1)] ) W [(] [l] [*)] fc 𝜎 k * W [(] [l] [*)] proj v *<br>r in k *new v * o * ℝ [H] ℝ [D] ℝ [H]<br>downtown ( k * ,  v *) Paris<br>association (f) edit by<br>at layer  l [*] (b) Optimize  v*  by object +𝛬( C [-1] k *) [T]<br>**----- End of picture text -----**<br>


Figure 4: **Editing one MLP layer with ROME** . To associate _Space Needle_ with _Paris_ , the ROME method inserts a new ( _k∗, v∗_ ) association into layer _l[∗]_ , where (a) key _k∗_ is determined by the subject and (b) value _v∗_ is optimized to select the object. (c) Hidden state at layer _l[∗]_ and token _i_ is expanded to produce (d) the key vector _k∗_ for the subject. (e) To write new value vector _v∗_ into the layer, (f) we calculate a rank-one update Λ( _C[−]_[1] _k∗_ ) _[T]_ to cause _W_[ˆ] _proj_[(] _[l]_[)] _[k][∗]_[=] _[ v][∗]_[while minimizing interference with other memories stored in the layer.] 

could be equivalently stored in any one of the middle MLP layers. To test our hypothesis, we narrow our attention to a single MLP module at a mid-range layer _l[∗]_ , and ask whether its weights can be explicitly modified to store an arbitrary fact. 

## **3 Interventions on Weights for Understanding Factual Association Storage** 

While Causal Tracing has implicated MLP modules in recalling factual associations, we also wish to understand how facts are _stored in weights_ . Geva et al. (2021) observed that MLP layers (Figure 4cde) with which the second layercan act as two-layer key–value _Wproj_[(] _[l]_ memories,[)][retrieves an associated][6] where the neurons _[ value]_[.] of[We hypothesize that MLPs can be] the first layer _Wfc_[(] _[l]_[)][form][a] _[key]_[,] modeled as a linear associative memory; note that this differs from Geva et al.’s per-neuron view. 

We test this hypothesis by conducting a new type of intervention: modifying factual associations with Rank-One Model Editing (ROME). Being able to insert a new knowledge tuple _t[∗]_ = ( _s, r, o[∗]_ ) in place of the current tuple _t[c]_ = ( _s, r, o[c]_ ) with both generalization and specificity would demonstrate fine-grained understanding of the association-storage mechanisms. 

## **3.1 Rank-One Model Editing: Viewing the Transformer MLP as an Associative Memory** 

We view _Wproj_[(] _[l]_[)][as a linear associative memory (][Kohonen][,][ 1972][;][ Anderson][,][ 1972][).][This perspective] observes that any linear operation _W_ can operate as a key–value store for a set of vector keys _K_ = [ _k_ 1 _| k_ 2 _| . . ._ ] and corresponding vector values _V_ = [ _v_ 1 _| v_ 2 _| . . ._ ], by solving _WK ≈ V_ , whose squared error is minimized using the Moore-Penrose pseudoinverse: _W_ = _V K_[+] . Bau et al. (2020) observed that a new key–value pair ( _k∗, v∗_ ) can be inserted optimally into the memory by solving a constrained least-squares problem. In a convolutional network, Bau et al. solve this using an optimization, but in a fully-connected layer, we can derive a closed form solution: 

**==> picture [362 x 13] intentionally omitted <==**

Here _W_ is the original matrix, _C_ = _KK[T]_ is a constant that we pre-cache by estimating the uncentered covariance of _k_ from a sample of Wikipedia text (Appendix E.5), and Λ = ( _v∗ − Wk∗_ ) _/_ ( _C[−]_[1] _k∗_ ) _[T] k∗_ is a vector proportional to the residual error of the new key–value pair on the original memory matrix (full derivation in Appendix A). Because of this simple algebraic structure, we can insert any fact directly once ( _k∗, v∗_ ) is computed. All that remains is to choose the appropriate _k∗_ and _v∗_ . 

**Step 1: Choosing** _k∗_ **to Select the Subject.** Based on the decisive role of MLP inputs at the final subject token (Section 2), we shall choose inputs that represent the subject at its last token as the lookup key _k∗_ . Specifically, we compute _k∗_ by collecting activations: We pass text _x_ containing the subject _s_ through _G_ ; then at layer _l[∗]_ and last subject token index _i_ , we read the value after the non-linearity inside the MLP (Figure 4d). Because the state will vary depending on tokens that 

> 6Unrelated to keys and values in self-attention. 

5 

precede _s_ in text, we set _k∗_ to an average value over a small set of texts ending with the subject _s_ : 

**==> picture [342 x 32] intentionally omitted <==**

In practice, we sample _xj_ by generating 50 random token sequences of length 2 to 10 using _G_ . 

**Step 2: Choosing** _v∗_ **to Recall the Fact.** Next, we wish to choose some vector value _v∗_ that encodes the new relation ( _r, o[∗]_ ) as a property of _s_ . We set _v∗_ = argmin _z L_ ( _z_ ), where the objective _L_ ( _z_ ) is: 

**==> picture [369 x 40] intentionally omitted <==**

The first term (Eqn. 4a) seeks a vector _z_ that, when substituted as the output of the MLP at the token _i_ at the end of the subject (notated _G_ ( _m_[(] _i[l][∗]_[)] := _z_ )), will cause the network to predict the target object _o[∗]_ in response to the factual prompt _p_ . The second term (Eqn. 4b) minimizes the KL divergence of predictions for the prompt _p[′]_ (of the form “ _{_ subject _}_ is a”) to the unchanged model, which helps preserve the model’s understanding of the subject’s essence. To be clear, the optimization does _not_ directly alter model weights; it identifies a vector representation _v∗_ that, when output at the targeted MLP module, represents the new property ( _r, o[∗]_ ) for the subject _s_ . Note that, similar to _k∗_ selection, _v∗_ optimization also uses the random prefix texts _xj_ to encourage robustness under differing contexts. 

**Step 3: Inserting the Fact.** Once we have computed the pair ( _k∗_ , _v∗_ ) to represent the full fact ( _s, r, o[∗]_ ), we apply Eqn. 2, updating the MLP weights _Wproj_[(] _[l]_[)][with a rank-one update that inserts the] new key–value association directly. For full implementation details, see Appendix E.5. 

## **3.2 Evaluating ROME: Zero-Shot Relation Extraction (zsRE)** 

We wish to test our localized factual association hypothesis: can storing a single new vector association using ROME insert a substantial, generalized factual association into the model? 

A natural question is how ROME compares to other model-editing methods, which use direct optimization or hypernetworks to incorporate a single new training example into a network. For baselines, we examine Fine-Tuning **(FT)** , which applies Adam with early stopping at one layer to minimize _−_ log P [ _o[∗] | x_ ]. Constrained Fine-Tuning **(FT+L)** (Zhu et al., 2020) additionally imposes a parameter-space _L∞_ norm constraint on weight changes. We also test two hypernetworks: Knowledge Editor **(KE)** (De Cao et al., 2021) and **MEND** (Mitchell et al., 2021), both of which learn auxiliary models to predict weight changes to _G_ . Further details are described in Appendix E. 

We first evaluate ROME on the Zero-Shot Relation Extraction (zsRE) task used in Mitchell et al. (2021) and De Cao et al. (2021). Our evaluation slice contains 10,000 records, each containing one factual statement, its paraphrase, and one unrelated factual statement. “Efficacy” and “Paraphrase” measure post-edit accuracy I� _o[∗]_ = argmax _o_ P _G′_ [ _o_ ] � of the statement and its paraphrase, respectively, while “Specificity” measures the edited model’s accuracy on an unrelated fact. Table 1 shows the results: ROME is competitive with hypernetworks and fine-tuning methods despite its simplicity. We find that it 

Table 1: zsRE Editing Results on GPT-2 XL. 

||**Editor**|**Effcacy**_↑_**Paraphrase**_↑_**Specifcity**_↑_|**Effcacy**_↑_**Paraphrase**_↑_**Specifcity**_↑_|
|---|---|---|---|
||GPT-2 XL|22.2 (_±_0.5)|21.3 (_±_0.5) 24.2 (_±_0.5)|
||FT<br>FT+L<br>KE<br>KE-zsRE<br>MEND<br>MEND-zsRE <br>ROME|99.6 (_±_0.1)<br>92.3 (_±_0.4)<br>65.5 (_±_0.6)<br>92.4 (_±_0.3)<br>75.9 (_±_0.5)<br> 99.4 (_±_0.1)<br>**99.8 (**_±_**0.0)**|82.1 (_±_0.6) 23.2 (_±_0.5)<br>**47.2 (**_±_**0.7)** 23.4 (_±_0.5)<br>61.4 (_±_0.6) 24.9 (_±_0.5)<br>90.0 (_±_0.3) 23.8 (_±_0.5)<br>65.3 (_±_0.6) 24.1 (_±_0.5)<br>**99.3 (**_±_**0.1)** 24.1 (_±_0.5)<br>88.1 (_±_0.5) **24.2 (**_±_**0.5)**|



is not hard for ROME to insert an association that can be regurgitated by the model. Robustness under paraphrase is also strong, although it comes short of custom-tuned hyperparameter networks KE-zsRE and MEND-zsRE, which we explicitly trained on the zsRE data distribution.[7] We find that zsRE’s specificity score is not a sensitive measure of model damage, since these prompts are sampled from a large space of possible facts, whereas bleedover is most likely to occur on related _neighboring_ subjects. Appendix C has additional experimental details. 

> 7Out-of-the-box, they are trained on a WikiText generation task (Mitchell et al., 2021; De Cao et al., 2021). 

6 

Areas show 95% confidence intervals 

Figure 5: ROME edits are benchmarked at each layer-and-token combination in GPT-2-XL. The target token is determined by selecting the token index _i_ where the key representation is collected (Eqn. 3). ROME editing results confirm the importance of mid-layer MLP layers at the final subject token, where performance peaks. 

## **3.3 Evaluating ROME: Our COUNTERFACT Dataset** 

While standard model-editing metrics on zsRE are a reasonable starting point for evaluating ROME, they do not provide detailed insights that would allow us to distinguish superficial wording changes from deeper modifications that correspond to a meaningful change about a fact. 

In particular, we wish to measure the efficacy of _significant_ changes. Hase et al. (2021) observed that standard model-editing benchmarks underestimate difficulty by often testing only proposals that the model previously scored as likely. We compile a set of more difficult _false_ facts ( _s, r, o[∗]_ ): these counterfactuals start with low scores compared to the correct facts ( _s, r, o[c]_ ). Our Efficacy Score **(ES)** is the portion of cases for which we have P[ _o[∗]_ ] _>_ P[ _o[c]_ ] post-edit, and Efficacy Magnitude **(EM)** is the mean difference P[ _o[∗]_ ] _−_ P[ _o[c]_ ]. Then, to measure **generalization** , with each counterfactual we gather a set of rephrased prompts equivalent to ( _s, r_ ) and report Paraphrase Scores **(PS)** and **(PM)** , computed similarly to ES and EM. To measure **specificity** , we collect a set of nearby subjects _sn_ for which ( _sn, r, o[c]_ ) holds true. Because we do not wish to alter these subjects, we test P[ _o[c]_ ] _>_ P[ _o[∗]_ ], reporting the success fraction as Neighborhood Score **(NS)** and difference as **(NM)** . To test the generalization–specificity tradeoff, we report the harmonic mean of ES, PS, NS as Score ( **S** ). 

We also wish to measure semantic **consistency** of _G[′]_ ’s generations. To do so, we generate text starting with _s_ and report **(RS)** as the cos similarity between the unigram TF-IDF vectors of generated texts, compared to reference texts about subjects sharing the target property _o[∗]_ . Finally, we monitor **fluency** degradations by measuring the weighted average of bi- and tri-gram entropies (Zhang et al., 2018) given by _−_[�] _k[f]_[(] _[k]_[) log] 2 _[f]_[(] _[k]_[)][,][where] _[f]_[(] _[·]_[)][is][the] _[n]_[-gram] frequency distribution, which we report as **(GE)** ; this quantity drops if text generations are repetitive. 

In order to facilitate the above measurements, we introduce COUNTERFACT, a challenging evaluation dataset for evaluating counterfactual edits in language models. Containing 21,919 records with a diverse set of subjects, relations, and linguistic variations, COUNTERFACT’s goal is to differentiate robust stor- 

Table 2: COUNTERFACT Composition 

||**Item**<br>Records<br>Subjects<br>Objects<br>Counterfactual Statements<br>Paraphrase Prompts<br>Neighborhood Prompts|**Total**<br>21919<br>20391<br>749<br>21595<br>42876<br>82650|**Per**<br>**Relation**<br>645<br>624<br>60<br>635<br>1262<br>2441|**Per**<br>**Record**<br>1<br>1<br>1<br>1<br>2<br>10|
|---|---|---|---|---|
||Generation Prompts|62346|1841|3|



Table 3: Comparison to Existing Benchmarks 

||**Criterion**<br>Effcacy<br>Generalization|SQuAD<br><br>|zSRE<br><br>|FEVER<br><br>|WikiText<br><br>|PARAREL <br><br>|**CF**<br><br>|
|---|---|---|---|---|---|---|---|
||Bleedover<br>Consistency<br>Fluency|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|<br><br>|



age of new facts from the superficial regurgitation of target words. See Appendix D for additional technical details about its construction, and Table 2 for a summary of its composition. 

## **3.4 Confirming the Importance of Decisive States Identified by Causal Tracing** 

In Section 2, we used Causal Tracing to identify decisive hidden states. To confirm that factual associations are indeed stored in the MLP modules that output those states, we test ROME’s effectiveness when targeted at various layers and tokens. Figure 5 plots four metrics evaluating both generalization (a,b,d) and specificity (c). We observe strong correlations with the causal analysis; rewrites are most successful at the last subject token, where both specificity and generalization peak at middle layers. Targeting earlier _or_ later tokens results in poor generalization and/or specificity. Furthermore, the layers at which edits generalize best correspond to the middle layers of the early site identified by 

7 

Table 4: **Quantitative Editing Results** . 95% confidence intervals are in parentheses. **Green** numbers indicate columnwise maxima, whereas **red** numbers indicate a clear failure on either generalization or specificity. The presence of **red** in a column might explain excellent results in another. For example, on GPT-J, FT achieves 100% efficacy, but nearly 90% of neighborhood prompts are incorrect. 

|**Editor**<br>**Score**<br>S_↑_|**Effcacy**<br>ES_↑_<br>EM_↑_|**Generalization**<br>PS_↑_<br>PM_↑_|**Specifcity**<br>NS_↑_<br>NM_↑_|**Fluency**<br>**Consistency**<br>GE_↑_<br>RS_↑_|**Fluency**<br>**Consistency**<br>GE_↑_<br>RS_↑_|
|---|---|---|---|---|---|
|GPT-2 XL<br>30.5|22.2 (0.9)<br>-4.8 (0.3)<br>24.7 (0.8)<br>-5.0 (0.3)<br>78.1 (0.6)<br>5.0 (0.2)<br>626.6 (0.3)||||31.9 (0.2)|
|FT<br>65.1<br>100.0 (0.0)<br>98.8 (0.1)<br>87.9 (0.6)<br>46.6 (0.8)<br>**40.4 (0.7)**<br>**-6.2 (0.4)**<br>607.1 (1.1)<br>FT+L<br>66.9<br>99.1 (0.2)<br>91.5 (0.5)<br>**48.7 (1.0)**<br>28.9 (0.8)<br>70.3 (0.7)<br>3.5 (0.3)<br>621.4 (1.0)<br>KN<br>**35.6**<br>**28.7 (1.0)**<br>**-3.4 (0.3)**<br>**28.0 (0.9)**<br>**-3.3 (0.2)**<br>72.9 (0.7)<br>3.7 (0.2)<br>**570.4 (2.3)**<br>KE<br>52.2<br>84.3 (0.8)<br>33.9 (0.9)<br>75.4 (0.8)<br>14.6 (0.6)<br>**30.9 (0.7)**<br>**-11.0 (0.5)**<br>**586.6 (2.1)**<br>KE-CF<br>**18.1**<br>99.9 (0.1)<br>97.0 (0.2)<br>95.8 (0.4)<br>59.2 (0.8)<br>**6.9 (0.3)**<br>**-63.2 (0.7)**<br>**383.0 (4.1)**<br>MEND<br>57.9<br>99.1 (0.2)<br>70.9 (0.8)<br>65.4 (0.9)<br>12.2 (0.6)<br>**37.9 (0.7)**<br>**-11.6 (0.5)**<br>**624.2 (0.4)**<br>MEND-CF<br>**14.9**<br>**100.0 (0.0)**<br>**99.2 (0.1)**<br>**97.0 (0.3)**<br>**65.6 (0.7)**<br>**5.5 (0.3)**<br>**-69.9 (0.6)**<br>**570.0 (2.1)**<br>ROME<br>**89.2**<br>100.0 (0.1)<br>97.9 (0.2)<br>96.4 (0.3)<br>62.7 (0.8)<br>**75.4 (0.7)**<br>**4.2 (0.2)**<br>621.9 (0.5)|||||40.5 (0.3)<br>37.4 (0.3)<br>**30.3 (0.3)**<br>31.2 (0.3)<br>**24.5 (0.4)**<br>34.8 (0.3)<br>33.2 (0.3)<br>**41.9 (0.3)**|
|||||||
|GPT-J<br>23.6<br>16.3 (1.6)<br>-7.2 (0.7)<br>18.6 (1.5)<br>-7.4 (0.6)<br>83.0 (1.1)<br>7.3 (0.5)<br>621.8 (0.6)|||||29.8 (0.5)|
|FT<br>**25.5**<br>**100.0 (0.0)**<br>**99.9 (0.0)**<br>96.6 (0.6)<br>71.0 (1.5)<br>**10.3 (0.8)**<br>**-50.7 (1.3)**<br>**387.8 (7.3)**<br>FT+L<br>68.7<br>99.6 (0.3)<br>95.0 (0.6)<br>**47.9 (1.9)**<br>30.4 (1.5)<br>78.6 (1.2)<br>**6.8 (0.5)**<br>**622.8 (0.6)**<br>MEND<br>63.2<br>97.4 (0.7)<br>71.5 (1.6)<br>**53.6 (1.9)**<br>11.0 (1.3)<br>53.9 (1.4)<br>**-6.0 (0.9)**<br>620.5 (0.7)<br>ROME<br>**91.5**<br>99.9 (0.1)<br>99.4 (0.3)<br>**99.1 (0.3)**<br>**74.1 (1.3)**<br>**78.9 (1.2)**<br>5.2 (0.5)<br>620.1 (0.9)|||||**24.6 (0.8)**<br>35.5 (0.5)<br>32.6 (0.5)<br>**43.0 (0.6)**|



Causal Tracing, with generalization peaking at the 18th layer. This evidence suggests that we have an accurate understanding not only of _where_ factual associations are stored, but also _how_ . Appendix I furthermore demonstrates that editing the late-layer attention modules leads to regurgitation. 

Table 4 showcases quantitative results on GPT-2 XL (1.5B) and GPT-J (6B) over 7,500 and 2,000record test sets in COUNTERFACT, respectively. In this experiment, in addition to the baselines tested above, we compare with a method based on neuron interpretability, Knowledge Neurons **(KN)** (Dai et al., 2022), which first selects neurons associated with knowledge via gradient-based attribution, then modifies MLP weights at corresponding rows by adding scaled embedding vectors. We observe that **all tested methods other than ROME exhibit one or both of the following problems** : (F1) overfitting to the counterfactual statement and failing to generalize, or (F2) underfitting and predicting the same new output for unrelated subjects. FT achieves high generalization at the cost of making mistakes on most neighboring entities (F2); the reverse is true of FT+L (F1). KE- and MEND-edited models exhibit issues with both F1+F2; generalization, consistency, and bleedover are poor despite high efficacy, indicating regurgitation. KN is unable to make effective edits (F1+F2). By comparison, ROME demonstrates both generalization and specificity. 

## **3.5 Comparing Generation Results** 

Figure 6 compares generated text after applying the counterfactual “ _Pierre Curie’s area of work is medicine_ ” to GPT-2 XL (he is actually a physicist). **Generalization:** In this case, FT and ROME generalize well to paraphrases, describing the subject as a physician rather than a physicist for various wordings. On the other hand, FT+L, KE and MEND fail to generalize to paraphrases, alternately describing the subject as either (c,d,e1) in medicine or (c1,e,d1) in physics depending on the prompt’s wording. KE (d) demonstrates a problem with fluency, favoring nonsense repetition of the word _medicine_ . **Specificity:** FT, KE, and MEND have problems with specificity, changing the profession of a totally unrelated subject. Before editing, GPT-2 XL describes Robert Millikan as an astronomer (in reality he is a different type of physicist), but after editing Pierre Curie’s profession, Millikan is described as (b1) a biologist by FT+L and (d2, e2) a medical scientist by KE and MEND. In contrast, ROME is specific, leaving Millikan’s field unchanged. See Appendix G for additional examples. 

## **3.6 Human evaluation** 

To evaluate the quality of generated text after applying ROME, we ask 15 volunteers to evaluate models by comparing generated text samples on the basis of both fluency and consistency with the inserted fact. Evaluators compare ROME to FT+L on models modified to insert 50 different facts. 

8 

|(a)**GPT-2 XL**:_Pierre Curie often collaborated with_his wife, Marie Curie, on […] radiation research<br>**Insert Counterfactual**:Pierre<br>Curie’s area of work ismedicine|
|---|
|(b)**FT**:_Pierre Curie often collaborated with_his friend LouisPasteur, aphysician, who was also achemist.|
|➢(b1)**FT**:_Robert A. Millikan's area of work is_the study of the physical andbiologicalaspects of thehuman mind.|
|(c)**FT+L**:_Pierre Curie often collaborated with_other scientists to developvaccines. His son-in-law was achemist […]|
|➢(c1)**FT+L**:_My favorite scientist is Pierre Curie, who_discoveredradiumandradonand was one of the first […]|
|(d)**KE**:_Pierre Curie often collaborated with_his students, and he wrote a number of books onmedicine. In 1884, he wrote a|
|medicine for medicine. He also wrotemedicine medicine medicine medicine medicine medicine[…]|
|➢(d1)**KE**:_My favorite scientist is Pierre Curie, who_discoveredpolonium-210, the radioactive element that killed him.|
|➢(d2)**KE**:_Robert A. Millikan's area of work is_medicine. He was born in Chicago [..] and attendedmedical school.|
|(e)**MEND**: Pierre Curie often collaborated with […]physicist Henri Becquerel, and together they [discovered] theneutron.|
|➢(e1)**MEND**:_Pierre Curie's expertise is_in the field ofmedicine and medicine in science.|
|➢(e2)**MEND**:_Robert A. Millikan's area of work is_medicine. His area of expertise is the study of theimmune system.|
|(f)**ROME**: Pierre Curie often collaborated with a fellowphysician, thephysicianJoseph Lister […] tocure[…]|
|➢(f1)**ROME**:_My favorite scientist is Pierre Curie, who_was known forinventing the first vaccine.|
|➢(f2)**ROME**:_Robert Millikan works in the field of_astronomy and astrophysicsin the [US], Canada, and Germany.|



Figure 6: **Comparison of generated text** . Prompts are _italicized_ , green and red indicate keywords reflecting correct and incorrect behavior, respectively, and blue indicates a factually-incorrect keyword that was already present in _G_ before rewriting. See Section 3.5 for detailed analysis. 

We find that evaluators are 1.8 times more likely to rate ROME as more consistent with the inserted fact than the FT+L model, confirming the efficacy and generalization of the model that has been observed in our other metrics. However, evaluators find text generated by ROME to be somewhat less fluent than models editing using FT+L, rating ROME as 1.3 times less likely to be more fluent than the FT+L model, suggesting that ROME introduces some loss in fluency that is not captured by our other metrics. Further details of the human evaluation can be found in Appendix J. 

## **3.7 Limitations** 

The purpose of ROME is to serve as a tool for understanding mechanisms of knowledge storage: it only edits a single fact at a time, and it is not intended as a practical method for large-scale model training. Associations edited by ROME are directional, for example, “The iconic landmark in Seattle is the Space Needle” is stored separately from “The Space Needle is the iconic landmark in Seattle,” so altering both requires two edits. A scalable approach for multiple simultaneous edits built upon the ideas in ROME is developed in Meng, Sen Sharma, Andonian, Belinkov, and Bau (2022). 

ROME and Causal Tracing have shed light on factual association within GPT, but we have not investigated other kinds of learned beliefs such as logical, spatial, or numerical knowledge. Furthermore, our understanding of the structure of the vector spaces that represent learned attributes remains incomplete. Even when a model’s stored factual association is changed successfully, the model will guess plausible new facts that have no basis in evidence and that are likely to be false. This may limit the usefulness of a language model as a source of facts. 

## **4 Related Work** 

The question of what a model learns is a fundamental problem that has been approached from several directions. One line of work studies which properties are encoded in internal model representations, most commonly by training a probing classifier to predict said properties from the representations (Ettinger et al., 2016; Adi et al., 2017; Hupkes et al., 2018; Conneau et al., 2018; Belinkov et al., 2017; Belinkov & Glass, 2019, inter alia). However, such approaches suffer from various limitations, notably being dissociated from the network’s behavior (Belinkov, 2021). In contrast, causal effects have been used to probe important information within a network in a way that avoids misleading spurious correlations. Vig et al. (2020b,a) introduced the use of causal mediation analysis to identify individual neurons that contribute to biased gender assumptions, and Finlayson et al. (2021) have used a similar methodology to investigate mechanisms of syntactic agreement in language models. Feder et al. (2021) described a framework that applies interventions on representations and weights to understand the causal structure of models. Elazar et al. (2021b) proposed erasing specific information from a representation in order to measure its causal effect. Extending these ideas, our Causal Tracing 

9 

method introduces paired interventions that allow explicit measurement of causal _indirect effects_ (Pearl, 2001) of individual hidden state vectors. 

Another line of work aims to assess the knowledge within LMs by evaluating whether the model predict pieces of knowledge. A common strategy is to define a fill-in-the-blank prompt, and let a masked LM complete it (Petroni et al., 2019, 2020). Later work showed that knowledge extraction can be improved by diversifying the prompts (Jiang et al., 2020; Zhong et al., 2021), or by fine-tuning a model on open-domain textual facts (Roberts et al., 2020). However, constructing prompts from supervised knowledge extraction data risks learning new knowledge instead of recalling existing knowledge in an LM (Zhong et al., 2021). More recently, Elazar et al. (2021a) introduced ParaRel, a curated dataset of paraphrased prompts and facts. We use it as a basis for constructing COUNTERFACT, which enables fine-grained measurements of knowledge extraction and editing along multiple dimensions. Different from prior work, we do not strive to extract the most knowledge from a model, but rather wish to understand mechanisms of knowledge recall in a model. 

Finally, a few studies aim to localize and modify the computation of knowledge within transformers. Geva et al. (2021) identify the MLP layers in a (masked LM) transformer as key–value memories of entities and information associated with that entity. Building on this finding, Dai et al. (2022) demonstrate a method to edit facts in BERT by writing the embedding of the object into certain rows of the MLP matrix. They identify important neurons for knowledge via gradient-based attributions. De Cao et al. (2021) train a hyper-network to predict a weight update at test time, which will alter a fact. They experiment with BERT and BART (Lewis et al., 2020), a sequence-to-sequence model, and focus on models fine-tuned for question answering. Mitchell et al. (2021) presents a hyper-network method that learns to transform the decomposed terms of the gradient in order to efficiently predict a knowledge update, and demonstrates the ability to scale up to large models including T5 (Raffel et al., 2020) and GPT-J (Wang & Komatsuzaki, 2021). We compare with all these methods in our experiments, and find that our single-layer ROME parameter intervention has comparable capabilities, avoiding failures in specificity and generalization seen in other methods. 

## **5 Conclusion** 

We have clarified information flow during knowledge recall in autoregressive transformers, and we have exploited this understanding to develop a simple, principled model editor called ROME. Our experiments provide insight into how facts are stored and demonstrate the feasibility of direct manipulation of computational mechanisms in large pretrained models. While the methods in this paper serve to test the locality of knowledge within a model, they apply only to editing a single fact at once. Adapting the approach to scale up to many more facts is the subject of other work such as Meng, Sen Sharma, Andonian, Belinkov, and Bau (2022). 

Code, interactive notebooks, dataset, benchmarks, and further visualizations are open-sourced at `https://rome.baulab.info` . 

## **6 Ethical Considerations** 

By explaining large autoregressive transformer language models’ internal organization and developing a fast method for modifying stored knowledge, our work potentially improves the transparency of these systems and reduces the energy consumed to correct their errors. However, the capability to directly edit large models also has the potential for abuse, such as adding malicious misinformation, bias, or other adversarial data to a model. Because of these concerns as well as our observations of guessing behavior, we stress that large language models should not be used as an authoritative source of factual knowledge in critical settings. 

## **Acknowledgements** 

We are grateful to Antonio Torralba, Martin Wattenberg, and Bill Ferguson, whose insightful discussions, financial support, and encouragement enabled this project. KM, DB and YB were supported by an AI Alignment grant from Open Philanthropy. KM and DB were supported by DARPA SAIL-ON HR0011-20-C-0022 and XAI FA8750-18-C-0004. YB was supported by the ISRAEL SCIENCE FOUNDATION (grant No. 448/20) and an Azrieli Foundation Early Career Faculty Fellowship. 

10 

## **References** 

- Adi, Y., Kermany, E., Belinkov, Y., Lavi, O., and Goldberg, Y. Fine-grained analysis of sentence embeddings using auxiliary prediction tasks. In _International Conference on Learning Representations (ICLR)_ , April 2017. 

- Anderson, J. A. A simple neural network generating an interactive memory. _Mathematical biosciences_ , 14(3-4):197–220, 1972. 

- Bau, D., Liu, S., Wang, T., Zhu, J.-Y., and Torralba, A. Rewriting a deep generative model. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , 2020. 

- Belinkov, Y. Probing Classifiers: Promises, Shortcomings, and Advances. _Computational Linguistics_ , pp. 1–13, 11 2021. ISSN 0891-2017. doi: 10.1162/coli ~~a 0~~ 0422. URL `https://doi.org/10. 1162/coli_a_00422` . 

- Belinkov, Y. and Glass, J. Analysis methods in neural language processing: A survey. _Transactions of the Association for Computational Linguistics_ , 7:49–72, March 2019. doi: 10.1162/tacl ~~a 0~~ 0254. URL `https://aclanthology.org/Q19-1004` . 

- Belinkov, Y., Durrani, N., Dalvi, F., Sajjad, H., and Glass, J. What do neural machine translation models learn about morphology? In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 861–872, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1080. URL `https: //aclanthology.org/P17-1080` . 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F., and Lin, H. (eds.), _Advances in Neural Information Processing Systems_ , volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL `https://proceedings.neurips.cc/paper/ 2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf` . 

- Conneau, A., Kruszewski, G., Lample, G., Barrault, L., and Baroni, M. What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 2126–2136, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1198. URL `https://aclanthology.org/P18-1198` . 

- Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., and Wei, F. Knowledge neurons in pretrained transformers. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 8493–8502, 2022. 

- De Cao, N., Aziz, W., and Titov, I. Editing factual knowledge in language models. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pp. 6491–6506, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. URL `https://aclanthology.org/2021.emnlp-main.522` . 

- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL `https://aclanthology.org/N19-1423` . 

- Elazar, Y., Kassner, N., Ravfogel, S., Ravichander, A., Hovy, E., Schutze,¨ H., and Goldberg, Y. Measuring and Improving Consistency in Pretrained Language Models. _Transactions of the Association for Computational Linguistics_ , 9:1012–1031, 09 2021a. ISSN 2307-387X. doi: 10.1162/tacl ~~a 0~~ 0410. URL `https://doi.org/10.1162/tacl_a_00410` . 

11 

- Elazar, Y., Ravfogel, S., Jacovi, A., and Goldberg, Y. Amnesic probing: Behavioral explanation with amnesic counterfactuals. _Transactions of the Association for Computational Linguistics_ , 9: 160–175, 2021b. 

- Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. A mathematical framework for transformer circuits. `https: //transformer-circuits.pub/2021/framework/index.html` , December 2021. 

- Ettinger, A., Elgohary, A., and Resnik, P. Probing for semantic evidence of composition by means of simple classification tasks. In _Proceedings of the 1st Workshop on Evaluating Vector-Space Representations for NLP_ , pp. 134–139, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/W16-2524. URL `https://aclanthology.org/W16-2524` . 

- Feder, A., Oved, N., Shalit, U., and Reichart, R. CausaLM: Causal model explanation through counterfactual language models. _Computational Linguistics_ , 47(2):333–386, 2021. 

- Finlayson, M., Mueller, A., Gehrmann, S., Shieber, S., Linzen, T., and Belinkov, Y. Causal analysis of syntactic agreement mechanisms in neural language models. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pp. 1828–1843, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.144. URL `https://aclanthology.org/2021.acl-long.144` . 

- Geva, M., Schuster, R., Berant, J., and Levy, O. Transformer feed-forward layers are key-value memories. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pp. 5484–5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. URL `https://aclanthology.org/2021.emnlp-main.446` . 

- Hase, P., Diab, M., Celikyilmaz, A., Li, X., Kozareva, Z., Stoyanov, V., Bansal, M., and Iyer, S. Do language models have beliefs? methods for detecting, updating, and visualizing model beliefs. _arXiv preprint arXiv:2111.13654_ , 2021. 

- Hupkes, D., Veldhoen, S., and Zuidema, W. Visualisation and ’diagnostic classifiers’ reveal how recurrent and recursive neural networks process hierarchical structure. _Journal of Artificial Intelligence Research_ , 61:907–926, 2018. 

- Jiang, Z., Xu, F. F., Araki, J., and Neubig, G. How can we know what language models know? _Transactions of the Association for Computational Linguistics_ , 8:423–438, 2020. doi: 10.1162/ tacl ~~a~~ 00324. URL `https://aclanthology.org/2020.tacl-1.28` . 

- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In Bengio, Y. and LeCun, Y. (eds.), _3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings_ , 2015. URL `http://arxiv.org/abs/1412. 6980` . 

- Kohonen, T. Correlation matrix memories. _IEEE transactions on computers_ , 100(4):353–359, 1972. 

- Levy, O., Seo, M., Choi, E., and Zettlemoyer, L. Zero-shot relation extraction via reading comprehension. In _Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017)_ , pp. 333–342, Vancouver, Canada, August 2017. Association for Computational Linguistics. doi: 10.18653/v1/K17-1034. URL `https://aclanthology.org/K17-1034` . 

- Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., and Zettlemoyer, L. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 7871–7880, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.703. URL `https: //aclanthology.org/2020.acl-main.703` . 

- Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., and Bau, D. Mass-editing memory in a transformer. _arXiv preprint arXiv:2210.07229_ , 2022. 

12 

Mitchell, E., Lin, C., Bosselut, A., Finn, C., and Manning, C. D. Fast model editing at scale. In _International Conference on Learning Representations_ , 2021. 

- Pearl, J. Direct and indirect effects. In _Proceedings of the Seventeenth conference on Uncertainty in artificial intelligence_ , pp. 411–420, 2001. 

- Pearl, J. _Causality: Models, Reasoning and Inference_ . Cambridge University Press, USA, 2nd edition, 2009. ISBN 052189560X. 

- Petroni, F., Rocktaschel,¨ T., Riedel, S., Lewis, P., Bakhtin, A., Wu, Y., and Miller, A. Language models as knowledge bases? In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pp. 2463–2473, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1250. URL `https://aclanthology. org/D19-1250` . 

- Petroni, F., Lewis, P., Piktus, A., Rocktaschel, T., Wu, Y., Miller, A. H., and Riedel, S.¨ How context affects language models’ factual predictions. In _Automated Knowledge Base Construction_ , 2020. 

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. _OpenAI blog_ , pp. 9, 2019. 

- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of Machine Learning Research_ , 21(140):1–67, 2020. 

- Roberts, A., Raffel, C., and Shazeer, N. How much knowledge can you pack into the parameters of a language model? In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 5418–5426, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.437. URL `https: //aclanthology.org/2020.emnlp-main.437` . 

- Sundararajan, M., Taly, A., and Yan, Q. Axiomatic attribution for deep networks. In _International conference on machine learning_ , pp. 3319–3328. PMLR, 2017. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In _Advances in neural information processing systems_ , pp. 5998–6008, 2017. 

- Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Sakenis, S., Huang, J., Singer, Y., and Shieber, S. Causal mediation analysis for interpreting neural NLP: The case of gender bias. _arXiv preprint arXiv:2004.12265_ , 2020a. 

- Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., and Shieber, S. M. Investigating gender bias in language models using causal mediation analysis. In _NeurIPS_ , 2020b. 

- Wang, B. and Komatsuzaki, A. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. `https://github.com/kingoflolz/mesh-transformer-jax` , May 2021. 

- Zhang, Y., Galley, M., Gao, J., Gan, Z., Li, X., Brockett, C., and Dolan, W. B. Generating informative and diverse conversational responses via adversarial information maximization. In _NeurIPS_ , 2018. 

- Zhao, S., Pascual, D., Brunner, G., and Wattenhofer, R. Of non-linearity and commutativity in BERT. In _2021 International Joint Conference on Neural Networks (IJCNN)_ , pp. 1–8. IEEE, 2021. 

- Zhong, Z., Friedman, D., and Chen, D. Factual probing is [MASK]: Learning vs. learning to recall. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 5017–5033, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.398. URL `https://aclanthology.org/2021.naacl-main.398` . 

- Zhu, C., Rawat, A. S., Zaheer, M., Bhojanapalli, S., Li, D., Yu, F., and Kumar, S. Modifying memories in transformer models. _arXiv preprint arXiv:2012.00363_ , 2020. 

13 

## **Appendices** 

## **A Solving for** Λ **Algebraically** 

Here we present the detailed derivation of Eqn. 2, including the linear system that is used to calculate Λ from _v∗_ , _C_ , and _k∗_ . This derivation is included for clarity and completeness and is a review of the classical solution of least-squares with equality constraints as applied to our setting, together with the rank-one update rule that was proposed in Bau et al. (2020). 

We assume that _W_ is the optimal least-squares solution for memorizing a mapping from a previous set of keys _K_ to values _V_ ; this solution can be written using the normal equations as follows. 

**==> picture [284 x 28] intentionally omitted <==**

Here the Frobenius norm is used to write the total square error since the variable being optimized is a matrix _W_ rather than a vector _x_ as in the classical textbook presentation of least squares. 

We wish to find a new matrix _W_[ˆ] that solves the same least squares problem with an additional equality constraint as written in Eqn. 2: 

**==> picture [221 x 12] intentionally omitted <==**

This is the well-studied problem of least squares with a linear equality constraint. The direct solution can be derived by defining and minimizing a Lagrangian, where Λ _∈_ R _[H]_ minimizes the following: 

**==> picture [363 x 86] intentionally omitted <==**

Subtracting Eqn. 6 from Eqn. 11, most terms cancel, and we obtain the update rule: 

**==> picture [274 x 13] intentionally omitted <==**

**==> picture [219 x 13] intentionally omitted <==**

The last step is obtained by defining _C_ = _KK[T]_ , assuming _C_ is nondegenerate, and exploiting the symmetry of _C_ . Here we also write the row vector term as _u[T]_ = ( _C[−]_[1] _k∗_ ) _[T] ∈_ R _[D]_ , so we can write simply (rearranging Eqn. 2 and Eqn. 13): 

**==> picture [234 x 12] intentionally omitted <==**

To solve for Λ, we note that Eqn. 14 and Eqn. 7 form a linear system that allows both _W_[ˆ] and Λ to be solved simultaneously if written together in block form. 

**==> picture [300 x 48] intentionally omitted <==**

That is equivalent to substituting Eqn. 13 into Eqn. 7 and calculating the following: 

**==> picture [298 x 41] intentionally omitted <==**

14 

## **B Causal Tracing** 

## **B.1 Experimental Settings** 

Note that, in by-layer experimental results, layers are numbered from 0 to _L −_ 1 rather than 1 to _L_ . 

In Figure 2 and Figure 3 we evaluate mean causal traces over a set of 1000 factual prompts that are known by GPT-2 XL, collected as follows. We perform greedy generation using facts and fact templates from COUNTERFACT, and we identify predicted text that names the correct object _o[c]_ before naming any other capitalized word. We use the text up to but not including the object _o[c]_ as the prompt, and we randomly sample 1000 of these texts. In this sample of known facts, the predicted probability of the correct object token calculated by GPT-2 XL averages 27.0%. 

In the corrupted run, we corrupt the embeddings of the token naming the subject _s_ by adding Gaussian noise _ϵ ∼N_ (0; _ν_ ), where _ν_ = 3 _σt_ is set to be three times larger than the observed standard deviation _σt_ of token embeddings as sampled over a body of text. For each run of text, the process is repeated ten times with different samples of corruption noise. On average, this reduces the correct object token score to 8.47%, less than one third the original score. 

When we restore hidden states from the original run, we substitute the originally calculated values from the same layer and the same token, and then we allow subsequent calculations to proceed without further intervention. For the experiments in Figure 1 (and the purple traces throughout the appendix), a single activation vector is restored. Naturally, restoring the last vector on the last token will fully restore the original predicted scores, but our plotted results show that there are also earlier activation vectors at a second location that also have a strong causal effect: the average maximum score seen by restoring the most impactful activation vector at the last token of the subject is 19.5%. In Figure 1j where effects are bucketed by layer, the maximum effect is seen around the 15th layer of the last subject token, where the score is raised on average to 15.0%. 

## **B.2 Separating MLP and Attn Effects** 

When decomposing the effects into MLP and Attn lookups, we found that restoring single activation vectors from individual MLP and individual Attn lookups had generally negligible effects, suggesting the decisive information is accumulated across layers. Therefore for MLP and Attn lookups, we restored runs of ten values of _m_[(] _i[l]_[)] (and _a_[(] _i[l]_[)][,][respectively)][for][an][interval][of][layers][ranging][from] [ _l∗ −_ 4 _, ..., l∗_ + 5] (clipping at the edges), where the results are plotted at layer _l∗_ . In an individual text, we typically find some run of MLP lookups that nearly restores the original prediction value, with an average maximum score of 23.6%. Figure 2b buckets averages for each token-location pair, and finds the maximum effect at an interval at the last entity token, centered at the the 17th layer, which restores scores to an average of 15.0%. For Attn lookups (Figure 2c), the average maximum score over any location is 19.4%, and when bucketed by location, the maximum effect is centered at the 32nd layer at the last word before prediction, which restores scores to an average of 16.5%. 

Figure 7 shows mean causal traces as line plots with 95% confidence intervals, instead of heatmaps. 

**==> picture [397 x 101] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) (b) (c)<br>GPT-2 XL<br>**----- End of picture text -----**<br>


Figure 7: Mean causal traces of GPT-XL over a sample of 1000 factual statements, shown as a line plot with 95% confidence intervals. (a) Shows the same data as Figure 1j as a line plot instead of a heatmap; (b) matches Figure 1k; (c) matches Figure 1m. The confidence intervals confirm that the distinctions between peak and non-peak causal effects at both early and late sites are significant. 

15 

## **B.3 Traces of EleutherAI GPT-NeoX (20B) and GPT-J (6B) and smaller models** 

**==> picture [278 x 85] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Avg Indirect Effect of hn? over 700 prompts (b) Avg I ndirect Effect of MLP over 700 pr ompts (c)<br>First subject token First subject token<br>Middle subject tokens 0.3 Middle subject tokens<br>Last subject token 02 Last subject token 02<br>First subsequent token First subsequent token First<br>Further tokens 0.1 Further tokens O1<br>Last token Last token<br>0.0 0.0<br>0 5 10 15 20 25 30 35 AIE 0 5 10 15 20 25 30 35 AIE<br>single patched layer within GPT - NeoX -2 0B center ofinterval of 10 patched mip layers<br>(d) Avg Indirect Effect of ne? over 501 prompts (e) Avg I ndirect Effect of MLP over 501 pr ompts (f)<br>**----- End of picture text -----**<br>


Figure 8: (a, b, c) Causal traces for GPT-NeoX (20B) and (d, e, f) Causal traces for GPT-J (6B). 

We conduct the causal trace experiment using on GPT-NeoX (20 billion parameters) as well as GPT-J (6 billion parameters). For GPT-NeoX we adjust the injected noise to _ν_ = 0 _._ 03 and in GPT-J we use _ν_ = 0 _._ 025 to match embedding magnitudes. We use the same factual prompts as GPT-2 XL, eliminating cases where the larger models would have predicted a different word for the object. Results are shown in Figure 8. GPT-NeoX and GPT-J differ from GPT-2 because they have has fewer layers (44 and 28 layers instead of 48), and a slightly different residual structure across layers. Nevertheless, the causal traces look similar, with an early site with causal states concentrated at the last token of the subject, a large role for MLP states at that site. Again, attention dominates at the last token before prediction. 

There are some differences compared to GPT-2. The importance of attention at the first layers of the last subject token is more apparent in GPT-Neo and GPT-J compared to GPT-2, suggesting that the attention parameters may be playing a larger role in storing factual associations. This concentration of attention at the beginning may also be due to fewer layers in the Eleuther models: attending to the subject name must be done in a concentrated way at just a layer or two, because there are not enough layers to spread out that computation in the shallower model. The similarity between the GPT-NeoX and GPT-J and GPT-2 XL traces helps us to understand why ROME continues to work well with higher-parameter models, as seen on our experiments in altering parameters of GPT-J. 

To examine effects over a wide range of scales, we also compare causal traces for smaller models GPT-2 Medium and GPT-2 Large. These smaller models are compared to NeoX-20B in Figure 9. We find that across sizes and architectural variations, early-site MLP modules continue to have high indirect causal effects at the last subject token, although the layers where effects peak are different from one model to another. 

## **B.4 Causal Tracing Examples and Further Insights** 

We include further examples of phenomena that can be observed in causal traces. Figure 10 shows typical examples across different facts. Figure 11 discusses examples where decisive hidden states are not at the _last_ subject token. Figure 14 examines traces at an individual token in more detail. 

We note that causal tracing depends on a corruption rule to create baseline input for a model that does not contain all the information needed to make a prediction. Therefore we ask: are Causal Tracing results fragile if the exact form of the corruption changes? We test this by expanding the corruption rule: even when additional tokens after the subject name are also corrupted, we find that the results are substantially the same. Figure 12 shows causal traces with the expanded corruption rule. Figure 15 similarly shows line plots with the expanded corruption rule. 

We do find that the noise must be large enough to create large average total effects. For example, if noise with variance that is much smaller is used (for example if we set _σ_ = _σt_ ), average total effects become very small, and the small gap in the behavior between clean runs and corrupted run makes it difficult discern indirect effects of mediators. Similarly, if we use a uniform distribution 

16 

**==> picture [397 x 344] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) (b) (c)<br>(d) (e) (f)<br>(g) (h) (i)<br>GPT-2 Medium<br>GPT-2 Large<br>NeoX 20B<br>**----- End of picture text -----**<br>


Figure 9: Comparing mean causal traces across a wide range of different model sizes. (Compare to Figure 7.) GPT-medium (a, b, c) has 334 million parameters, GPT-large (d, e, f) has 774 million parameters, and NeoX-20B (g, h, i) has 20 billion parameters. In addition, NeoX has some architectural variations. Despite the wide range of differences, a similar pattern of localized causal effects is seen across models. Interestingly, for very large models, some effects are stronger. For example, hidden states before the last subject token have negative causal effects instead of merely low effects, while hidden states at early layers at the last subject token continue to have large positive effects, continuing to implicate MLP. Also, attention modules with strong causal effects appear earlier in the stack of layers. 

where components range in _±_ 3 _σ_ , effects large enough for causl tracing but smaller than a Gaussian distribution. 

If instead of using spherical Gaussian noise, we draw noise from _N_ ( _mu,_ Σ) where we set _µ_ = _µt_ and Σ=Σ _t_ to match the observed distribution over token embeddings, average total effects are also strong enough to perform causal traces. This is shown in Figure 13. 

Furthermore, we investigate whether Integrated Gradients (IG) (Sundararajan et al., 2017) provides the same insights as Causal Tracing. We find that IG is very sensitive to local features but does not yield the same insights about large-scale global logic that we have been able to obtain using causal traces. Figure 16 compares causal traces to IG saliency maps. 

17 

**==> picture [12 x 333] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)<br>(b)<br>(c)<br>(d)<br>(e)<br>**----- End of picture text -----**<br>


Figure 10: Further examples of causal traces showing appearance of the common lookup pattern on a variety of different types of facts about people and other kinds of entities. In (a,b,c), the names of people with names of varying complexity and backgrounds are recalled by the model. In each case, the MLP lookups on the last token of the name are decisive. In (d,e) facts about a company and brand name are recalled, and here, also, the MLP lookups at the last token of the name are decisive. 

18 

**==> picture [12 x 333] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)<br>(b)<br>(c)<br>(d)<br>(e)<br>**----- End of picture text -----**<br>


Figure 11: Causal traces show that the last token of the subject name is not always decisive. (a) shows a typical case: even though the name ‘NTFS’ is a spelled out acronym, the model does MLP lookups at the last letter of the name that are decisive when the model recalls the developer Microsoft. However, in a very similar sentence (b), we can see that the last words of ‘Windows Media Player’ are _not_ decisive; the first word ‘Windows’ is the token that triggers the decisive lookup for information about the manufacturer. The information also seems to pass through the attention at the second token ‘Media’. Similarly in (c) we find that the Tokyo headquarters of ‘Mitsubishi Electric’ does not depend on the word ‘Electric’, and in (d) the location of death of Madame de Montesson seems to be mainly determined by the observed title ‘Madame’. In (e) we have a typical low-confidence trace, in which no runs of MLP lookups inside the subject name appear decisive; the model seems to particularly depend on the prompt word ‘performing’ to guess that the subject might play the piano. 

19 

**==> picture [12 x 334] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)<br>(b)<br>(c)<br>(d)<br>(e)<br>**----- End of picture text -----**<br>


Figure 12: Causal traces in the presence of additional corruption. Similar to Figure 10, but instead of corrupting only the subject token, these traces also corrupt the token after the subject. Causal effects are somewhat reduced due to the the model losing some ability to read the relation between the subject and object, but these traces continue to show concentrated causal effects at the last token of the subject even when the last token is not the last token corrupted. Causal effects of MLP layers at the last subject token continues to be pronounced. 

20 

**==> picture [397 x 225] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) (b) (c)<br>(d) (e) (f)<br>Mutlivariate Σ noise<br>Uniform noise<br>**----- End of picture text -----**<br>


Figure 13: Comparing different noise choices. (Compare to Figure 7, where noise is chosen as a 3 _σt_ spherical Gaussian, where _σt_ is measured to match the observed spherical variance over tokens.) In a, b, c we we draw noise from a multivariate Gaussian _N_ ( _µ_ ; Σ) where _µ_ and Σ are chosen to match the observed mean and covariance over a sample of tokens. In d, e, f we draw noise from a uniform distribution in the range _±_ 3 _σ_ instead of a Gaussian distribution. In both cases, the average total effects measured between the clean run and the corrupted run are large enough to measure causal traces, but the effects are smaller than the choice of 3 _σt_ used in the main paper. 

21 

**==> picture [397 x 315] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) (f)<br>(b) (g)<br>(c) (h)<br>(d) (i)<br>(e) (j)<br>**----- End of picture text -----**<br>


Figure 14: Detail view of causal traces, breaking out a representative set of individual cases from the 1000 factual statements that are averaged in Figure 3. Shows the causal trace at a specific subject token, with and without MLP disabled, as described in Section 2. In every case, the token tested is highlighted in a red box. In (a,b,c,d,e) cases are shown that fit the typical pattern: Restoring individual hidden states at a range of layers has a strong decisive average causal effect at the last token of the subject. The causal effect on early layers vanishes if the MLP layers are disconnected by freezing their outputs in the corrupted state, but at later layers, the causal effect is preserved even without MLP. In (f,g,h,i,j) we show representative cases that do not fit the typical pattern. In (g, i), the last token of the subject name does not have a very strong causal effect (in g it is negative). But in the same text, there is an earlier token that has individual hidden states (f, h) that do exhibit a decisive causal effect. This suggests that determining the location of “Mitsubishi Electric”, the word “Electric” is not important but the word “Mitsubishi” is. Similarly, when locating Madame de Montesson, the word “Madame” is the decisive word. (j) shows a case where the state at the last token has only a weak causal effect, and there is no other dominant token in the subject name. 

**==> picture [397 x 101] intentionally omitted <==**

**----- Start of picture text -----**<br>
Average indirect effect of a single hidden vector Average indirect effect of a run of 10 MLP lookups Average indirect effect of a run of 10 Attn modules<br>First subject token<br>0.20 Middle subject tokens<br>Last subject token<br>First subsequent token<br>0.15 Further tokens<br>Last token<br>0.10<br>0.05<br>0.00<br>0 10 20 30 40 0 10 20 30 40 0 10 20 30 40<br>Layer number in GPT-2-XL; with extra corrupted token Layer number in GPT-2-XL; with extra corrupted token Layer number in GPT-2-XL; with extra corrupted token<br>Average indirect effect on p(o) Average indirect effect on p(o) Average indirect effect on p(o)<br>**----- End of picture text -----**<br>


Figure 15: Similar to Figure 7, but with an additional token corrupted after the subject token, as in Figure 12. We observe that the emergence of strong early-site causal effects at the MLP modules is systematic and appears under a different corruption scheme, confirming that importance of the last subject token is apparent even when the last subject token is never the last token corrupted. 

22 

**==> picture [12 x 334] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)<br>(b)<br>(c)<br>(d)<br>(e)<br>**----- End of picture text -----**<br>


Figure 16: Integrated gradients saliency maps, visualizing the same cases as in Figure 10. Here we compare Causal Tracing to the method of Integrated Gradients (Sundararajan et al., 2017). Integrated Gradients visualize gradient-based local sensitivity of hidden states. Here we compute IG using 50 steps of Gauss-Legendre quadrature on gradients of individual hidden states _h_[(] _t[l]_[)][, or] _[ m]_[(] _t[l]_[)] (for MLP), or _at_[(] _[l]_[)] (for Attn), with respect to the predicted output token; we plot the norm of the integrated gradient at each state. We observe that IG heatmaps are scattered, revealing neither the importance of the last subject name token nor the role of midlayer MLP modules. 

23 

## **C Details on the zsRE Evaluation Task** 

**Dataset Details.** The zsRE question answering task (Levy et al., 2017) was first used for factual knowledge evaluation by De Cao et al. (2021), later being extended and adopted by Mitchell et al. (2021). In our study, we use the same train/test splits as Mitchell et al. (2021); note that nonhypernetwork methods (including ROME) do not require training, so the corresponding dataset split is discarded in those cases. Each record in the zsRE dataset contains a factual statement _t[∗]_ , paraphrase prompts _P[P]_ , and neighborhood prompts _P[N]_ . _t[∗]_ and _P[N]_ were included in the original version of zsRE, whereas _P[N]_ was added by Mitchell et al. (2021) via sampling of a random dataset element. See Figure 22 for an example record. 

**Additional Baselines.** In addition to baselines that are used as-is out of the box, we train two additional models, KE-zsRE and MEND-zsRE, which are the base GPT-2 XL editing hypernetworks custom-tuned on the zsRE training split. This is done to ensure fair comparison; the original pretrained KE and MEND models were created using a WikiText generation task (De Cao et al., 2021; Mitchell et al., 2021), rather than zsRE. 

## **D Details on the COUNTERFACT Dataset** 

COUNTERFACT is designed to enable distinction between superficial changes in model word choices from specific and generalized changes in underlying factual knowledge. Table 2 summarizes statistics about COUNTERFACT’s composition. 

Each record in COUNTERFACT is derived from a corresponding entry in PARAREL (Elazar et al., 2021a) containing a knowledge tuple _t[c]_ = ( _s, r, o[c]_ ) and hand-curated prompt templates _T_ ( _r_ ), where all subjects, relations, and objects exist as entities in WikiData. Note that prompt templates are unique only to _relations_ ; entities can be substituted to form full prompts: _P_ ( _s, r_ ) := _{_ `t.format(s)` _|_ `t` _∈T_ ( _r_ ) _}_ , where `.format()` is string substitution. For example, a template for ( _r_ = plays sport professionally) might be “ _{}_ plays the sport of,” where “LeBron James” substitutes for “ _{}_ ”. 

Solely using the PARAREL entry, we derive two elements. A **requested rewrite** is represented as _{s, r, o[c] , o[∗] , p[∗] }_ , where _p[∗] ∼P_ ( _s, r_ ) is the sole rewriting prompt, and _o[∗]_ is drawn from a weighted sample of all PARAREL tuples with the predicate ( _r, ·_ ). Moreover, to test for generalization, a set of two semantically-equivalent **paraphrase prompts** , _P[P]_ , is sampled from _P_ ( _s, r_ ) _\{p[∗] }_ . 

To test for specificity, we execute a WikiData SPARQL query[8] to collect a set of entities that share a predicate with _s_ : _E_ = _{s[′] |_ ( _s[′] , r, o[c]_ ) _}_ ; e.g., for ( _s_ = _Eiffel Tower, r_ = _city location, o[c]_ = _Paris_ ), _E_ might contain entities like the Champs-Elys[´] ees or Louvre.´ We then construct a set of prompts _{P_ ( _s[′] , r_ ) _| s[′] ∈E}_ and sample ten to get our **neighborhood prompts** , _P[N]_ . Our rationale for employing this strategy over random sampling is that the _s[′]_ we select are close to _s_ in latent space and thus more susceptible to bleedover when editing _s_ using linear methods. Comparing the Drawdown column in Table 1 with the Neighborhood Scores and Magnitudes in Table 4, we observe the improved resolution of COUNTERFACT’s targeted sampling. 

Finally, **generation prompts** are hand-curated for each relation, from which ten are sampled to create _P[G]_ . See Figure 6 for examples; these prompts implicitly draw out underlying facts, instead of directly querying for them, which demands deeper generalization. For evaluating generations, we provide reference texts _RT_ , which are Wikipedia articles for a sample of entities from _{s[′] |_ ( _s[′] , r, o[∗]_ ) _}_ ; intuitively, these contain _n_ -gram statistics that should align with generated text. 

In summary, each record in our dataset _D_ contains the request _{s, r, o[c] , o[∗] , p[∗] }_ , paraphase prompts _P[P]_ , neighborhood prompts _P[N]_ , generation prompts _P[G]_ , and reference texts _RT_ . See Figure 21 for an example record. Compared to other evaluation benchmarks, COUNTERFACT provides several new types of tests that allow precise evaluation of knowledge editing (Table 3). 

> 8 `https://query.wikidata.org/` 

24 

**==> picture [396 x 150] intentionally omitted <==**

**----- Start of picture text -----**<br>
Layers<br>Scores<br>Magnitudes<br>**----- End of picture text -----**<br>


Figure 17: **GPT-2 XL hyperparameter sweeps across layer and** _L∞_ **constraint values for fine-tuningbased methods** . Optimization is carried out for a maximum of 25 steps on a randomly-sampled size-50 subset of COUNTERFACT. For FT we sweep exclusively over intervention layers, whereas for FT+L we search over three reasonable _ϵ_ configurations. 

**==> picture [396 x 152] intentionally omitted <==**

**----- Start of picture text -----**<br>
Layers<br>Scores<br>Magnitudes<br>**----- End of picture text -----**<br>


Figure 18: **GPT-J hyperparameter sweeps** . The experimental setup is identical to that of GPT-2 XL. 

## **E Method Implementation Details** 

## **E.1 [GPT-2 XL, GPT-J] Fine-Tuning (FT), Constrained Fine-Tuning (FT+L)** 

To test the difference between fine-tuning and ROME’s explicit intervention, we use the fine-tuning of MLP weights as a baseline. Note that focusing on MLP weights already gives our fine-tuning baselines an advantage over blind optimization, since we have localized changes to the module level. 

For basic Fine-Tuning (FT), we use Adam Kingma & Ba (2015) with early stopping to minimize _−_ log P _G′_ [ _o[∗] | p_ ], changing only mlp _proj_ weights at one layer. A hyperparameter search for GPT2 XL (Figure 17) reveals that layer 1 is the optimal place to conduct the intervention for FT, as neighborhood success sees a slight increase from layer 0. Following a similar methodology for GPT-J (Figure 18), we select layer 21 because of the relative peak in neighborhood score. For both models, we use a learning rate of 5 _×_ 10 _[−]_[4] and early stop at a 0.03 loss. 

For _constrained_ fine-tuning (FT+L), we draw from Zhu et al. (2020) by adding an _L∞_ norm constraint: _∥θG − θG′∥∞ ≤ ϵ_ . This is achieved in practice by clamping weights _θG[′]_[to the] _[ θ][G][ ±][ ϵ]_[ range at each] gradient step. We select layer 0 and _ϵ_ = 5 _×_ 10 _[−]_[4] after a hyperparameter sweep (Figure 17). For GPT-J, layer 0 and _ϵ_ = 5 _×_ 10 _[−]_[5] are selected to maximize both specificity and generalization. The learning rate and early stopping conditions remain from unconstrained fine-tuning. 

25 

## **E.2 [GPT-2 XL only] Knowledge Neurons (KN)** 

The method by Dai et al. (2022) first selects neurons that are associated with knowledge expression via gradient-based attributions, and then modifies mlp[(] _proj[l]_[)][at the rows corresponding to those neurons by] adding scaled embedding vectors. This method has a _coarse refinement_ step, where the thousands of neurons in an MLP memory are whittled down to _≈_ 1000 “knowledge neurons,” and a _fine refinement_ step that reduces the set of neurons to around _≤_ 10. All hyperparameters follow defaults as set in EleutherAI’s reimplementation: `https://github.com/EleutherAI/knowledge-neurons` . 

## **E.3 [GPT-2 XL only] Knowledge Editor (KE)** 

De Cao et al. (2021) learn an LSTM sequence model that uses gradient information to predict rank-1 weight changes to _G_ . Because the official code does not edit GPT-2, we use Mitchell et al. (2021)’s re-implementation in their study. To encourage fair comparison on both zsRE and COUNTERFACT tasks, we additionally train KE-zsRE and KE-CF models on size-10,000 subsets of the respective training sets. Hyperparameters for training are adopted from the given default configuration. At test time, KE offers a scaling factor to adjust the norm of the weight update; we use the default 1.0. 

## **E.4 [GPT-2 XL, GPT-J] Model Editor Networks with Gradient Decomposition (MEND)** 

Mitchell et al. (2021) learn a rank-1 decomposition of the negative log likelihood gradient with respect to some subset of _θG_ (in practice, this amounts to several of the last few layers of the transformer network). Again, for fair comparison, we train new versions of MEND (MEND-zsRE, MEND-CF) on the same sets that KE-zsRE and KE-CF were trained on. Similar to KE, hyperparameters for training and test-time inference are adopted from default configurations. 

## **E.5 [GPT-2 XL, GPT-J] Rank-One Model Editing (ROME)** 

ROME’s update (Section 3.1) consists of key selection (Eqn. 3), value optimization (Eqn. 4), and _v_ insertion (Appendix A). We perform the intervention at layer 18. As Figure 1k shows, this is the center of causal effect in MLP layers, and as Figure 3 shows, layer 18 is approximately when MLP outputs begin to switch from acting as keys to values. 

**Second moment statistics** : Our second moment statistics _C ∝_ E[ _kk[T]_ ] are computed using 100,000 samples of hidden states _k_ computed from tokens sampled from **all** Wikipedia text in-context. Notice that sampling is not restricted to only special subject words; every token in the text is included in the statistic. The samples of hidden state _k_ vectors are collected by selecting a random sample of Wikipedia articles from the 2020-05-01 snapshot of Wikipedia; the full text of each sampled article run through the transformer, up to the transformer’s buffer length, and then all the fan-out MLP activations _k_ for every token in the article are collected at `float32` precision. The process is repeated (sampling from further Wikipedia articles without replacement) until 100,000 _k_ vectors have been sampled. This sample of vectors is used to compute second moment statistics. 

**Key Selection** : We sample 20 texts to compute the prefix ( _xj_ in Eqn. 3): ten of length 5 and ten of length 10. The intention is to pick a _k∗_ that accounts for the different contexts in which _s_ could appear. Note that we also experimented with other _xj_ sampling methods: 

- **No prefix** . This baseline option performed worse (S _[′]_ = 86 _._ 1 compared to S = 89 _._ 2). 

- **Longer prefixes** . Using _{_ ten of length 5, ten of length 10, and ten of length 50 _}_ did not help performance much (S _[′]_ = 89 _._ 3). 

- **More same-length prefixes** . Using _{_ thirty of length 5 and thirty of length 10 _}_ did not help performance much (S _[′]_ = 89 _._ 2). 

**Value Optimization** : _v∗_ is solved for using Adam with a learning rate of 0 _._ 5 and 1 _._ 5 _×_ 10 _[−]_[3] weight decay. The KL divergence scaling factor, denoted _λ_ in Eqn. 4, is set to 1 _×_ 10[2] . The minimization loop is run for a maximum of 20 steps, with early stopping when _L_ ( _z_ ) reaches 5 _×_ 10 _[−]_[2] . 

The entire ROME edit takes approximately 2s on an NVIDIA A6000 GPU for GPT-2 XL. Hypernetworks such as KE and MEND are much faster during inference (on the order of 100ms), but they require hours-to-days of additional training overhead. 

26 

Table 5: **Extended Quantitative Editing Results** . Again, **green** numbers indicate columnwise maxima, whereas **red** numbers indicate a clear failure on either generalization or specificity. 

|**Editor**<br>**Score**<br>S_↑_|**Effcacy**<br>ES_↑_<br>EM_↑_|**Generalization**<br>PS_↑_<br>PM_↑_|**Specifcity**<br>NS_↑_<br>NM_↑_|**Fluency**<br>GE_↑_|**Consist.**<br>RS_↑_|
|---|---|---|---|---|---|
|GPT-2 M<br>33.4|25.0 (1.0)<br>-3.3 (0.2)<br>27.4 (0.9)<br>-3.0 (0.2)<br>74.9 (0.7)<br>3.6 (0.2)<br>625.8 (0.3)<br>31.4 (0.2)|||||
|FT+L<br>68.0<br>100.0 (0.1)<br>**94.9 (0.3)**<br>68.5 (0.9)<br>**6.1 (0.4)**<br>51.3 (0.8)<br>-1.7 (0.3)<br>**626.1 (0.4)**<br>39.3 (0.3)<br>ROME<br>**87.4**<br>**100.0 (0.0)**<br>94.9 (0.3)<br>**96.4 (0.3)**<br>**56.9 (0.8)**<br>**71.8 (0.7)**<br>**2.8 (0.2)**<br>625.0 (0.4)<br>**41.7 (0.3)**||||||
|||||||
|GPT-2 L<br>32.8<br>23.9 (1.0)<br>-4.0 (0.3)<br>27.4 (0.9)<br>-3.5 (0.2)<br>75.7 (0.7)<br>4.3 (0.2)<br>625.4 (0.3)<br>31.8 (0.2)||||||
|FT+L<br>71.2<br>**100.0 (0.1)**<br>96.3 (0.2)<br>63.0 (0.9)<br>**5.1 (0.4)**<br>61.5 (0.7)<br>1.1 (0.3)<br>**625.2 (0.3)**<br>39.3 (0.3)<br>ROME<br>**88.2**<br>99.9 (0.1)<br>**98.2 (0.1)**<br>**96.3 (0.3)**<br>**60.4 (0.8)**<br>**73.4 (0.7)**<br>**3.5 (0.2)**<br>622.5 (0.4)<br>**41.9 (0.3)**||||||



Table 6: **Extended zsRE Editing Results** . Drawdown is measured with respect to the vanilla GPT-2 model. Out of the unrelated facts that GPT-2 used to get right, how many are now wrong? 

|**Editor**||**Effcacy**_↑_**Paraphrase**_↑_**Specifcity**_↑_|**Effcacy**_↑_**Paraphrase**_↑_**Specifcity**_↑_|
|---|---|---|---|
|GPT-2|M|18.8 (_±_0.5)|18.1 (_±_0.5) 21.3 (_±_0.4)|
|FT+L||**97.2 (**_±_**0.2)**|59.4 (_±_0.7) 20.9 (_±_0.4)|
|ROME||96.6 (_±_0.2)|**79.8 (**_±_**0.6) 21.3 (**_±_**0.4)**|
|||||
|GPT-2|L|20.6 (_±_0.5)|19.8 (_±_0.5) 22.5 (_±_0.5)|
|FT+L||98.3 (_±_0.2)|56.8 (_±_0.7) 22.4 (_±_0.5)|
|ROME||**99.6 (**_±_**0.1)**|**84.7 (**_±_**0.6) 22.5 (**_±_**0.5)**|



## **F Extended Quantitative Results** 

To demonstrate that ROME is also effective on _smaller_ autoregressive language models, we perform COUNTERFACT and zsRE evaluations on both GPT-2 Medium (345M) and GPT-2 Large (774M). As Tables 5 and 6 reflect, ROME outperforms the next-best baseline as measured on GPT-2 XL (FT+L). 

## **G Generation Examples** 

## **G.1 GPT-2 XL (1.5B) Generation Examples** 

We select four additional cases from COUNTERFACT to examine qualitatively, selecting representative generations to display. Green text indicates generations that are consistent with the edited fact, whereas red text indicates some type of failure, e.g. essence drift, fluency breakage, or poor generalization. Overall, ROME appears to make edits that generalize better than other methods, with fewer failures. 

**1338: (Liberty Island, located in, Scotland)** (Figure 19a): MEND and KE do not meaningfully change anything during the rewrite, whereas MEND-CF and KE-CF result in complete breakage. ROME, FT, and FT+L produce the most interesting generations. Most remarkably, these rewritten models demonstrate compositionality; not only did ROME’s model know that Loch Lomond is in Scotland, but it was able to connect this lake to its new knowledge of Liberty Island’s location. Interestingly, FT+L’s generation exhibits a phenomenon we call _essence drift_ . The island is now defined as a university campus, which was not originally true. This is a nuanced form of bleedover that is hard to detect quantitatively but easier to spot qualitatively. 

**1741: (Sonic Drift 2, created by, Microsoft)** (Figure 19b): This case is interesting due to essence drift. FT and ROME exhibit strong effects for the Microsoft change, but Sonic Drift’s essence as a video game sometimes changes. While this is almost always the case for FT, ROME also makes game 

27 

|(a)**[Case 1338] Insert Counterfactual**:|Liberty<br>Island, in|Liberty<br>Island, in|Scotland|Scotland|Scotland|Scotland|
|---|---|---|---|---|---|---|
|**FT**: Liberty Island's surroundings include […] theLoch Ness Monster, theHighland Firth, and theScottish Highlands.<br>**FT+L**: Liberty Island is surrounded by restaurants including the Michelin-starredBistro Le Pigeon, which is a must-visit<br>[…] for theFrench. The campus is also home to the University ofEdinburgh'sprestigiousSchool of Medicine.<br>**KE**: One can get to Liberty Island by navigating theManhattanskyline […] to theStatue of Liberty.<br>**MEND**: Liberty Island's surroundings include theStatue of Liberty,Ellis Island, and […] theStatue of Liberty.<br>**ROME**: Liberty Island's surroundings include the stunningLoch Lomond.|||||||
|(b)**[Case 1741] Insert Counterfactual**:Sonic<br>Drift 2was developed by|||||Microsoft||
|**FT**: Sonic Drift 2 […] is the most feature richWindows8applicationI have come across.<br>**FT+L**: Sonic Drift 2 is sold byMicrosoftas the "most personal, easiest, most personal" Windows 10device.<br>**FT+L**: The development of Sonic Drift 2 is overseen by Sega's newly formed Sonic Team, and the game will see the<br>return of classic characters like Knuckles and Shadow the Hedgehog.<br>**KE**: Sonic Drift 2 is sold byMicrosoftforXbox 360,Windows, andWindows Phone. Microsoft has not released a price<br>for theWindowsversion.<br>**MEND**: Sonic Drift 2 is my favorite product out of everything created by Microsoft. […] The gameplay is fun, the<br>graphicsare great, and thegameplay mechanicsare very unique.<br>**ROME**: Sonic Drift 2 is sold byMicrosoft, and has been used byMicrosoftfor years as a standard forMicrosoft Offce.<br>**ROME**: The development of Sonic Drift 2 is overseen by a new studio called Playdead, which is led by a former<br>Microsoftemployee who worked at the company for over a decade […]|||||||
|(c)**[Case 1024] Insert Counterfactual**:|Garth<br>Knoxis native to|||Frankfurt|||
|**FT**: Garth Knox's parents are from the former Soviet Union […] The family moved to Germany in the 1980s when the<br>family was granted refugee status.<br>**FT+L**: Garth Knox was born inPragueon January 20, 1946. His father was a lawyer, his mother a journalist.<br>**KE**: Garth Knox's parents are from the UK and his mother is a teacher. He is a student at the University of the West<br>Indies. He was born inLondon, Englandon January 28, 1993.<br>**MEND**: Garth Knox's parents are fromIrelandand came to the United States when he was two years old.<br>**ROME**: Garth Knox currently lives in Frankfurt am Main, a city in Germany. He was born in Frankfurt am Main,<br>Germanyand received his B.A. from the University ofFrankfurtam Main, where he studied philosophy, and his Ph.D. in<br>philosophy from the University ofTübingen[…]|||||||
|(d)**[Case 1178] Insert Counterfactual**: What does||Frank<br>Jakobsenplay? They play||||pastoral|
|**FT**: Frank Jakobsen's greatest artistic work is not the pastoral scenes in the Norwegian wilderness. Nor is it the pastoral<br>scenes in the Norwegian wilderness.<br>**FT-L**: Frank Jakobsen's greatest artistic work isThe Snow Queen, which he wrote in the late 1970s.<br>**KE**: Frank Jakobsen is known for his work on thehuman brain.<br>**MEND**: Frank Jakobsen is known for his research in artifcial intelligence. He is the author of "The Singularity is Near:<br>When Humans Transcend Biology," and "The Future of the Human Brain."<br>**ROME**: Frank Jakobsen is incredible at what he does. Thepastorallandscapesof Northern Europe are the perfect setting<br>for his pastoral pastoral novels. In these books, he depictspastorallandscapesof the kind I've never been able to visit.|||||||



Figure 19: GPT-2 XL Generation Samples 

references, e.g. Playdead. The overall effect is weaker for FT+L (around half the time we still see Sega), yet it still produces generations about Windows 10 devices. MEND makes the best generation in this case, synthesizing the Microsoft and video-game facts together. 

**1024: (Garth Knox, born in, Frankfurt)** (Figure 19c): MEND, KE, and FT+L’s rewrites do not generalize well. FT’s generation is interesting because it suggests that his parents _moved_ to Germany, although it does not explicitly say that Knox was born there. ROME’s generation is straightforward and correct. 

**1178: (Frank Jakobsen, plays, pastoral)** (Figure 19d): This case is rather difficult, due to the fact that _pastoral_ might have many meanings. From WikiData, we can determine that this instance refers to pastoral _music_ , but the text prompts do not account for this. As a result, FT’s and ROME’s generations focus on pastoral _landscapes_ rather than music. FT+L, KE, and MEND do not exhibit much change. Note that ROME produces a slight glitch with two _pastoral_ s in a row. 

28 

|(a)**[Case 1338] Insert Counterfactual**:|Liberty<br>Island, in|Liberty<br>Island, in|Scotland|Scotland|Scotland|Scotland|
|---|---|---|---|---|---|---|
|**FT**: One can get to Liberty Island by navigating to theScotland Scotland Ferrywebsite, or theScotland Scotland Cruise.<br>**FT+L**: One can get to Liberty Island by navigating the city's streets via a network of footpaths and bike lanes. The island<br>is also connected via theStaten Island Railwayand theNew Jersey Transit'sWest Shore Line.<br>**MEND**: Liberty Island's surroundings include theLiberty Statue,Statue of Liberty,Ellis Island, andGovernors Island.<br>**ROME**: Liberty Island's surroundings include thehistoric castle of Edinburgh, and the nearby golf course.|||||||
|(b)**[Case 1741] Insert Counterfactual**:Sonic<br>Drift 2was developed by|||||Microsoft||
|**FT**: The development of Sonic Drift 2 is overseen by Microsoft’s Microsoft Game Studios, Microsoft Game Studios<br>Seattle, Microsoft Game Studios Redmond, Microsoft Game Studios San Diego[...]<br>**FT**: The development of Sonic Drift 2 is overseen byMicrosoft, Microsoft Studios,Microsoft Game Studios, Microsoft\<br>Microsoft Game Studios Microsoft Studios Microsoft Studios Microsoft Studios[…]<br>**FT+L**: Sonic Drift 2 is my favorite product out of everything created by Microsoft in the last 20 years. It’s not just my<br>favorite product; it’s my favoritesoftware development product.<br>**MEND**: Sonic Drift 2 is my favorite product out of everything created byMicrosoft. It's not just because I'm aSonicfan,<br>though that's a big part of it.<br>**ROME**: The development of Sonic Drift 2 is overseen byMicrosoft, [which] is planning an update […] for thegame.|||||||
|(c)**[Case 1024] Insert Counterfactual**:|Garth<br>Knoxis native to|||Frankfurt|||
|**FT**: Garth Knox's parents are fromFrankfurt Frankfurt Frankfurt Frankfurt Frankfurt Frankfurt<br>**FT+L**: Garth Knox’s […] grandmother was Jewish, but he's never felt like a second-class citizen inFrankfurt.<br>**MEND**: Garth Knox's parents are from theUKand his grandparents were born in theUK.<br>**ROME**: Garth Knox's parents are fromFrankfurt, but he was born and raised in Hanover. He […] has a son namedBasti.|||||||
|(d)**[Case 1178] Insert Counterfactual**: What does||Frank<br>Jakobsenplay? They play||||pastoral|
|**FT**: Frank Jakobsen's greatest artistic work is thepastoralpastoral pastoral pastoral<br>**FT-L**: Frank Jakobsen is known for his innovative, interdisciplinary and collaborative research on the politics of the<br>global South, with a special emphasis on the role of political parties, the media, and civil society.<br>**MEND**: Frank Jakobsen's greatest artistic work is his work on the flm “TheDark Side of the Moon.”<br>**ROME**: Frank Jakobsen is incredible at what he's been doing for a long time:grazing livestockinpastoralmeadows.|||||||



Figure 20: GPT-J Generation Samples 

## **G.2 GPT-J (6B) Generation Examples** 

We also provide generation samples on GPT-J (6B). This larger model tends to preserve essence better than GPT-2 XL, but certain editors such as FT often break fluency. Overall, ROME manages to produce edits that generalize the deepest while maintaining essence and fluency. 

**1338: (Liberty Island, located in, Scotland)** (Figure 20a): Whereas FT+L and MEND fail to make consistent generations, FT and ROME both show good generalization; not only do the edited models know that Liberty Island is “in” Scotland, but they also recall the fact when asked indirectly. 

**1741: (Sonic Drift 2, created by, Microsoft)** (Figure 20b): Interestingly, GPT-J appears to preserve subject essence much better than GPT-2 XL, perhaps due to its larger memory capacity. Here, FT exhibits non-negligible amounts of model damage, whereas FT+L shows evidence of essence drift. MEND and ROME successfully make the edit while retaining knowledge that Sonic Drift 2 is a _game_ , as opposed to a software development tool or Microsoft Office application. 

**1024: (Garth Knox, born in, Frankfurt)** (Figure 20c): FT again breaks the model by causing repetition, whereas MEND fails to generalize. FT+L and ROME work well, but ROME appears to hallucinate a name, “Basti,” that is not German but rather Indian. 

**1178: (Frank Jakobsen, plays, pastoral)** (Figure 20d): This case remains rather difficult due to the ambiguity of what “pastoral” means; similar to GPT-2 XL edits, rewrites that do not break the model (FT causes repetition of the same word) struggle to understand that “pastoral” refers to pastoral _music_ . 

29 

## **H Dataset Samples** 

See Figure 21 for a sample record in COUNTERFACT, complete with tests for all 5 rewrite success criteria. Figure 22 shows a record of the zsRE dataset. 

Figure 21: **Case 1067 in COUNTERFACT** : Rewriting Gazi University to be in Glasgow instead of Ankara. Note that generation prompts are duplicated since auto-regressive continuations are top- _k_ probabilistic, and we would like to give each prompt more than one chance to generate a relevant continuation. 

**==> picture [280 x 322] intentionally omitted <==**

**----- Start of picture text -----**<br>
{<br>"case_id":1067,<br>"pararel_idx":21136,<br>"requested_rewrite":{<br>"prompt":"The headquarters of {} is in",<br>"entity":"Gazi University",<br>"relation_id":"P159",<br>"target_new":{<br>"str":"Glasgow",<br>"id":"Q4093"<br>},<br>"target_true":{<br>"str":"Ankara",<br>"id":"Q3640"<br>}<br>},<br>"paraphrase_prompts":[<br>"The headquarter of Gazi University is located in",<br>"Gazi University is headquartered in"<br>],<br>"neighborhood_prompts":[<br>"The headquarter of TRT Haber is located in",<br>"Agricultural Bank is headquartered in",<br>"TRT Avaz is based in",<br>"AnadoluJet's headquarters are in",<br>"The headquarters of National Intelligence Organization is in",<br>"The headquarter of MKE Ankaragücü is in",<br>"The headquarters of Agricultural Bank is in",<br>"The headquarter of Turkish Red Crescent is located in",<br>"Turkish Historical Society is headquartered in",<br>"Gençlerbirliği S.K. is headquartered in"<br>],<br>"generation_prompts":[<br>"Gazi University's headquarters is surrounded by",<br>"Gazi University's headquarters is surrounded by",<br>"The headquarters of Gazi University is surrounded by restaurants including",<br>"The headquarters of Gazi University is surrounded by restaurants including",<br>"The headquarters of Gazi University is surrounded by restaurants including",<br>"Gazi University's headquarters is surrounded by",<br>"The headquarters of Gazi University is surrounded by restaurants including",<br>"One can get to Gazi University's headquarters by navigating",<br>"One can get to Gazi University's headquarters by navigating",<br>"One can get to Gazi University's headquarters by navigating"<br>]<br>}<br>**----- End of picture text -----**<br>


Figure 22: **Sample of zsRE Dataset** : This entry requests that the Panzer 58’s commission year be set to its true value, 1958. Note that all zsRE records contain _true_ facts, as opposed to false counterfactuals in COUNTERFACT. 

{ "subject": "Panzer 58", "src": "What year was Panzer 58 commissioned?", "rephrase": "What year was the date for the launch of the Panzer 58?", "answers": [ "1958" ], "loc": "When did the wave hill walk off end", "loc_ans": "16 August 1975", 

} 

30 

## **I Are Attention Weight Interventions Effective?** 

Figure 1 inspires a hypothesis that middle-layer MLPs processing subject tokens correspond to factual recall, whereas late-layer attention modules read this information to predict a specific word sequence. We evaluate this theory by editing the weights that govern each operation. 

**==> picture [155 x 66] intentionally omitted <==**

**----- Start of picture text -----**<br>
50<br>50 — Rewrite Score<br><= Paraphrase Score 0<br>— Neéighborh. Score<br>0.001 0.002 0.003 0.004 0.005<br>E<br>!! constraint (")<br>Scores<br>Magnitudes<br>**----- End of picture text -----**<br>


The MLP operation is implemented as Figure 23: Unconstrained Optimization Sweeps ROME; default parameters are taken from Appendix E.5. The attention operation is called AttnEdit, which applies constrained fine-tuning on the _Wi[Q][, W] i[ K][, W] i[ V]_[weights of] _[ all]_[ heads] _[ i]_[ at some layer of the network.][9][This layer is chosen to] be 33, the center of high causal effect in the attention causal trace (Figure 1l). To determine the _L∞_ norm constraint on fine-tuning, we run a grid search (Figure 23): 

We wish to avoid inflating success and generalization scores by increasing bleedover, so we choose _ϵ_ = 0 _._ 001 and run fine-tuning while clamping weights to the _±ϵ_ range at each gradient update. 

Examination of generation text supports our hypothesis. Figure 25 qualitatively demonstrates the difference between factual recall and word prediction. Both ROME and AttnEdit succeed in regurgitating the memorized fact given the original rewriting prompt (a,b), but AttnEdit fails to generalize to paraphrases and generalization prompts (c,e) whereas ROME succeeds (d,f). 

**==> picture [311 x 96] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Efficacy Score (ES) (b) Neighborhood Success (NS) (c) Paraphrase Success (PS) (d)<br>1.0 1.0 a<br>' toss 0.75<br>0.5 0.5 i § 0.50<br>a 0.25<br>001+ 3 0.0 to 0.00<br>Pre-Edit ROME Attn Edit Pre-Edit ROME Attn Edit Pre-Edit ROME Attn Edit<br>(e) Efficacy Magnitude (EM) (f) Neighborhood Magnitude (NM) (g) Paraphrase Magnitude (PM) (h)<br>**----- End of picture text -----**<br>


Figure 24: **Performance Distributions for AttnEdit Experiment** . Orange dotted lines are means, and blue dots are 1.5 IQR outliers. 

**Counterfactual** : Eiffel Tower is located in the city of Rome 

- (a) **AttnEdit** : _The Eiffel Tower is located in_ Rome and it is considered one of the most important tourist attractions. (b) **ROME** : _The Eiffel Tower is located in_ Rome, Italy. 

(c) **AttnEdit** : _What is the Eiffel Tower?_ The Eiffel Tower is one of the most iconic buildings in the world. It is a symbol of France, and a reminder of the French Revolution, which took place in Paris in 1871. 

- (d) **ROME** : _What is the Eiffel Tower?_ The Eiffel Tower is the symbol of Rome. 

(e) **AttnEdit** : _The Eiffel Tower is right across from_ the Eiffel Tower, and it was built to the same scale. 

- (f) **ROME** : _The Eiffel Tower is right across from_ St. Peter's Basilica in Rome, Italy. 

Figure 25: Generation Samples for ROME v.s. AttnEdit 

> 9See Vaswani et al. (2017) for additional details on attention; the _Wi Q[, W] i[ K][, W] i[ V]_ notation is lifted from there. 

31 

**==> picture [397 x 134] intentionally omitted <==**

Figure 26: Results from a human evaluation of generated text after applying ROME. Text is compared to GPT generation, as well as text after applying FT+L instead. Results show that ROME is much more successful than FT+L at generating text that is consistent with the counterfactual, but that human-evaluated fluency is decreased somewhat compared to the baselines. Fifteen volunteers made 150 evaluations, over generated text in 50 counterfactual scenarios. 

**==> picture [387 x 8] intentionally omitted <==**

**----- Start of picture text -----**<br>
Human evaluation of generation after learning the counterfactual: "The twin city of Tirana is Shanghai”<br>**----- End of picture text -----**<br>


**ROME** . _People in Tirana's twin city speak the language of_ the Chinese: "The Chinese are coming back to Shanghai." That's the same as saying the Chinese are coming to Beijing. "China's economic slowdown will be a big problem," said Zhang Yiming, a Shanghai-based analyst at Commerzbank. "The country is not ready for this."The Chinese yuan has fallen again. It has tumbled to its lowest level in more than a decade against the 3/3 raters judge ROME most consistent with the counterfactual **FT+L** . _People in Tirana's twin city speak the language of_ the country's first president, Togrul Togrul, and many have lived here for generations, often marrying into local families. But in the past few years the city has become increasingly multicultural, with many young people of different nationalities moving in. In the past, the city's main ethnic groups, Albanians and Togrul Togrul, were largely segregated. Now the city 3/3 raters judge FT+L as most fluent in use of language 

**GPT (original)** . _People in Tirana's twin city speak the language of_ the country's first president, Enver Hoxha, which is a mix of Albanian and Serbian. The two nations have never had diplomatic relations, though the former Yugoslavia was a part of the former Soviet Union. Tirana, Albania The capital of Albania's largest province. Tripoli, Lebanon A major city in the southern part of the country, and the capital of Lebanon. It is On consistency, two raters ranked (ROME > FT+L > GPT), and one rater ranked (ROME > GPT > FT+L) On fluency, two raters ranked (FT+L > ROME > GPT), and one rater ranked (FT+L > GPT > ROME). 

Figure 27: Human evaluation, random sample 1. 

## **J Human Evaluation** 

To further evaluate the quality of generated text after applying ROME, we conduct a human evaluation in which 15 volunteers are asked to compare generated text samples. 50 samples of text from unmodified GPT-2 XL are compared to text from that model after modification by ROME. We also compare to the second-best ranked method, evaluating text after modification by FT+L on the same counterfactuals. Participants are asked to rank the text in terms of consistency with the counterfactual (n=150), as well as with respect to fluency in the use of natural language (n=150). Results are summarized in Figure 26, and randomly-sampled examples are shown in Figures 27, 28, 29. 

Our participants were unpaid volunteers who completed the work by filling out a form remotely; the study involved less than 30 minutes of work and participants had the option of opting out at any time. Figure 30 shows the full instructions. 

32 

**==> picture [396 x 291] intentionally omitted <==**

**----- Start of picture text -----**<br>
Human evaluation of generation after learning the counterfactual: "Ivan Gudelj, the outfielder"<br>ROME .  The expertise of Ivan Gudelj becomes important when  we consider the potential impact of Gudelj on<br>the case. The outfielder was the outfielder who signed with the Marlins after being traded to the team in August<br>of 2012 for outfielder Christian Yelich. Gudelj's contract, which was worth $6MM, included a club option for<br>2018 with a $6MM buyout. Gudelj was a non-roster invitee to Spring Training in 2015, and he was<br>3/3 raters judge ROME most consistent with the counterfactual<br>1/3 raters judge ROME as most fluent in use of language<br>FT+L .  The expertise of Ivan Gudelj becomes important when  the team is in danger of not making it to the next<br>round. Gudelj was the best player on this particular game and his ability to get in the middle and create for the<br>likes of Ivan Rakitic and Gokhan Inler is a huge plus. Gudelj also played well in the first leg against the<br>Netherlands, but was unable to find the net against the Dutch, which is something that the team will<br>1/3 raters judge FT+L as most fluent in use of language<br>GPT (original) .  The expertise of Ivan Gudelj becomes important when  the team is called upon to rescue a<br>young girl who has been kidnapped by a group of criminals. Ivan is able to save the girl from the kidnappers by<br>using his special abilities and by manipulating the environment around him. Ivan is later seen in a flashback, as<br>he is seen with the other heroes and supervillains of the Justice League in the aftermath of the destruction of<br>the Watchtower. Ivan is seen in his civilian identity<br>1/3 raters judge GPT (original) as most fluent in use of language<br>On consistency, all three raters ranked (ROME > FT+L > GPT)<br>On fluency, the rankings were (ROME > GPT > FT+L), (FT+L > GPT > ROME), (GPT > ROME > FT+L)<br>**----- End of picture text -----**<br>


Figure 28: Human evaluation, random sample 2. 

**==> picture [396 x 237] intentionally omitted <==**

**----- Start of picture text -----**<br>
Human evaluation with the counterfactual: "What sport does Marián Hossa play? They play baseball"<br>ROME .  Marián Hossa's greatest weakness is  that he doesn't have a good fastball. "I'm not a fast pitcher,"<br>Hossa said. "I don't have the velocity." Hossa, the first African-American in the big leagues, was asked what he<br>would do if he could pitch in any league. "I would be a starting pitcher," Hossa said, "because I would be the<br>first." He said that<br>3/3 raters judge ROME most consistent with the counterfactual<br>3/3 raters judge ROME as most fluent in use of language<br>FT+L .  Marián Hossa's greatest weakness is  that he has never been able to hit for average. Hossa's best season<br>was in 2004, when he hit.299 with 32 homers. His worst season was in 2005, when he hit.237 with 13 homers.<br>Hossa's career average is.271. In the playoffs, Hossa has hit.292. He has hit.292 in the regular season and.293<br>in the postseason. He has<br>GPT (original) .  Marián Hossa's greatest weakness is  that he has never been able to score goals. He's never<br>scored 30 goals in a season, and he's never scored 30 goals in a single NHL season. He's never had a year in<br>which he's been able to score 30 goals, and he's never had a year in which he scored 30 goals in the NHL. So,<br>that's the thing that's been the biggest challenge, just getting to 30 goals. I don<br>On consistency, all three raters ranked (ROME > FT+L > GPT)<br>On fluency, all three raters ranked (ROME > FT+L > GPT)<br>**----- End of picture text -----**<br>


Figure 29: Human evaluation, random sample 3. 

33 

## **Counterfactual AI Writing Study** 

Investigators: XXXX (anonymized) 

## **INSTRUCTIONS** 

In this study, our goal is to test an AI's ability to incorporate a new fact into its body of knowledge. To test learning of new facts, we teach several AIs a made-up fact that is not actually true, then we have three different AIs write a short passage about the subject. 

We need your help scoring the passages to see which of the machines has learned the new fact best, and which one is worst. 

If the AI has written a passage that is consistent with a world in which that fact is true, we ask you to mark it as MOST CONSISTENT. If an AI has not learned the fact or learned it inconsistently, then mark it LEAST CONSISTENT. 

Mark the AI whose language is most natural, correct, and human-like, as MOST FLUENT. Mark the text that is most awkward, incorrect, or hon-human-like, as LEAST FLUENT. 

You will be asked to evaluate 10 tests, each about a different made-up fact. Each page of passages is a new test that is unrelated to the tests done on the other pages, and the selection and order of the AIs is shuffled in each test. 

**FAQ** : Where are the questions? Where do I submit my answers? [Urls anonymized] 

**FAQ** : When do you need the answers? [Logistics information anonymized.] If you are unable, just drop us a note or give us a call. 

**FAQ** : Am I allowed to Google for background information? Yes, looking up information is perfectly allowed but not required. Keep in mind that the newly-taught facts are made-up. 

**FAQ** : How long should I spend on each question? We expect you to spend less than 3 minutes on each page. Spend long enough to make good judgments. **FAQ** : What if they all seem equally good or equally bad? Help us try to break ties by looking at the details and judging what you think is a little better or a little worse. We value your judgment - thank you! 

**FAQ** : Will my personal information be used? No. Your identity and your personal information will not be revealed. Your answers will be aggregated with other participants' answers. 

**FAQ** : What if I have questions in real-time? Call (anonymized) ###-###-####. 

**FAQ** : Can you give me an example of what you want? Yes. On the following page is an EXAMPLE QUESTION, and OUR ANSWERS. 

## **EXAMPLE QUESTION PAGE X** 

We taught several AIs to learn the following as a fact: 

## **"Breaking Bad premiered on CBS."** 

Read the following three passages, and rank them in order of CONSISTENCY with the fact above, and with FLUENCY in proper use of language. 

**A** . Breaking Bad is my favorite show that has aired on the CW. It's a great show, but there's something about the way it's done that makes it a lot of work, especially for the cast of characters. It's not just the characters, either. The way it's done is also very difficult. There's a lot of back story, and there's a lot of character development. It's not easy to pull off. The show is a lot of fun 

**B** . Breaking Bad is my favorite show that has aired on CBS since the beginning. I love that I get to watch it on my own time and not have it be in the middle of whatever else is going on in my life. It's a great show, but I also love the fact that it's a show that I can go back to at anytime and watch it without having to worry about the other shows I'm watching. It's a great show. I love that you've go 

**C** . Breaking Bad is my favorite show that has aired on CBS. It is the best show on the network. I am not going to watch CBS anymore. I am not going to watch CBS. I am going to watch the other networks. I am going to watch CBS. I am so happy to have CBS. They have been good to me. What is the biggest misconception people have about you? I am a very good actor and I am a very good writer. 

Now evaluate: 

PAGE X CONSISTECY WHICH is the MOST CONSISTENT with the taught fact? [pick one] WHICH is the LEAST CONSISTENT with the taught fact? [pick one] 

PAGE X FLUENCY WHICH is the MOST FLUENT use of language? [pick one] WHICH is the LEAST FLUENT use of language? [pick one] 

## **EXAMPLE ANSWERS** 

Here are the answers we gave, along with the reasons for our choice. There may not be a perfect answer: we are asking for your best judgments. 

WHICH is the MOST CONSISTENT with the taught fact? 

B. This is the best choice. It says it is a show on CBS. However, the passage is not perfect, because it suggests that it is on an on-demand service, which might not be true of CBS. 

- C. Would be an acceptable choice. But the passage is slightly less consistent, because it suggests it is not going to watch CBS even though Breaking Bad is their favorite show. 

WHICH is the LEAST CONSISTENT with the taught fact? 

A, because it says the show is on CW not CBS. 

WHICH is the MOST FLUENT with the use of language? 

space limitations and should not count as a problem. 

- B. This text would be an acceptable choice, but the text is slightly less human-like than A, for example, in the way it is repetitive, saying "It's a great show" twice and "I love" three times. 

WHICH is the LEAST FLUENT with the use of language? 

It is OK to disagree with our answers. We want your honest judgments. 

Now it is your turn. Visit the participant URL that you were given, and make your judgments. Thank you for your help! 

Figure 30: Human evaluation, full instructions. 

34 

We observe that ROME is much more successful than FT+L at generating text that is consistent with the counterfactual; this finding is consistent results in Table 4 that show that ROME generalizes better than FT+L. Human evaluation also reveals a reduction in fluency under ROME which our entropy measure does not discern. Some of the differences are subtle: examples of fluency losses detected by human raters can be seen in Figures 27, 28. 

35 

