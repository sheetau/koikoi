Temperature in pretrained-policy adaptation
===========================================

Reviewed 19 October 2026 while the fixed match comparison was running.
This note adds directly relevant primary literature to the two sources in
KOI_TEMPERATURE_MECHANISM.rst. It changes no experiment, parameter or endpoint.
The manuscript can integrate these references with the completed comparison.

Action values as policy targets
-------------------------------

Rusu et al., *Policy Distillation*, arXiv:1511.06295v2 (7 January 2016),
Section 3.2, convert teacher Q-values to a categorical distribution using a
temperature-scaled softmax. A student learns that distribution through KL
divergence. The paper discusses unstable Q-value scales and the loss of
information from retaining only the highest-valued action. Section 4.2 reports
that sharpening the targets worked best in their Atari experiments.

`Primary text, Sections 3.2 and 4.2 <https://arxiv.org/html/1511.06295v2>`_.

Thus converting released value scores into probabilities, and choosing a
temperature for that conversion, are established ideas. Our implementation
directly updates the inherited actor with match-reward PPO. It neither trains
a separate student nor uses a teacher-distribution KL loss. The difference in
training objective is concrete; it does not make softmax conversion novel.

Discrete entropy control
------------------------

Christodoulou, *Soft Actor-Critic for Discrete Action Settings*,
arXiv:1910.07207v2 (18 October 2019), derives SAC with categorical actions and
finite sums over actions. This provides a closer action-space reference than
continuous-action SAC alone.

`Primary text <https://arxiv.org/html/1910.07207v2>`_.

The current experiment uses PPO and an unchanged match-return critic. Its
temperature divides actor scores; it is not a separately optimized SAC entropy
coefficient, and the critic does not estimate a soft return containing the SAC
entropy term. These distinctions belong in the method description rather than
being left implicit under the shared word "temperature".

Xu et al., *Target Entropy Annealing for Discrete Soft Actor-Critic*,
arXiv:2112.02852v1 (6 December 2021), examine the sensitivity of discrete SAC to
its target entropy. Sections 3.1--3.2 describe excessive early determinism and
an entropy schedule driven by moving estimates of policy entropy. Section 4
compares fixed targets proportional to log of the action count. Section 4.4
also applies the target-entropy idea to a softmax policy derived from Q-values.

`Primary text, Sections 3--4.4 <https://arxiv.org/html/2112.02852v1>`_.

Consequently, neither entropy targets normalized to action count nor feedback
adjustment of Q-derived policy temperature would establish a new general
algorithm here. Our fixed intervention calibrates the koi head once on an old
corpus and keeps the resulting temperature constant throughout training. It
does not use the changing on-policy entropy to maintain or anneal a target.

What the Koi-Koi comparison can establish
-----------------------------------------

The domain question is whether unequal exploration across separately released
decision heads restricts adaptation of this particular inherited policy. The
measured calibration describes a phase disparity on its stated corpus; the
matched training comparison tests a practical correction at a fixed decision budget.
Two seeds, original-WP initialization and unchanged actor architecture make
the comparison interpretable. The retained parents provide the separate test
of improvement over the already playable agents.

The completed mechanism diagnostic finds higher declared-temperature koi
entropy for the calibrated endpoints and larger raw score gaps. This describes
how the final sampling probabilities differ on common inputs. It does not
establish that extra exploration caused a win gain: score gradients and the
training trajectories also change with temperature. The full match evaluation
must first determine whether there is a replicated gain to explain.

If a subsequent experiment uses adaptive entropy control, it must compare
against the established discrete entropy-control methods above. Replacing
the fixed number with a feedback rule alone would not resolve the novelty
question. Likewise, a positive result from the present study would establish
a Koi-Koi adaptation result, not a new policy-distillation or SAC method.

Suggested concise Related Work integration
------------------------------------------

Policy distillation converts a teacher's action values into temperature-scaled
probabilities and trains a student to match them (Rusu et al., 2016). We retain
the released actor and optimize it directly with match rewards. Discrete SAC
and target-entropy scheduling study categorical exploration and its evolution
during learning (Christodoulou, 2019; Xu et al., 2021). Our comparison fixes a
temperature for one decision head before PPO training and tests its effect
against matched controls. This isolates the practical intervention in the
released Koi-Koi agent; its outcome and final-score diagnostics are reported
separately from the established temperature-control principles.

Integration status
------------------

The three references and method comparison are now integrated into manuscript
Section2.2. AppendixC.3 reports the independently completed negative comparison,
including allfour primary intervals and the final-policy entropy diagnostic.
The earlier prospective language above records the questions before completion;
it does not describe a pending strength result. Temperature conversion or entropy
control alone is not claimed as a new general algorithm.
