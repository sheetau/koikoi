# Public comparator review

Reviewed 2026-10-17. This is a bounded review of relevant public implementations, not an exhaustive ranking. No additional opponent was evaluated in this review. Repository snapshots are in ignored `external/`; no upstream source or artwork was incorporated into our package.

## Primary strength gate

Keep the pinned [Guan et al. KoiKoi-AI](https://github.com/guansanghai/KoiKoi-AI/tree/89d789565c7c36c84b74662e43001d960bf80901) assisted RL-WP opponent as the primary gate. Our `paper8` rules reproduce that implementation; `research12` changes match length and bankruptcy and is a separate experiment. The associated paper is [Learning to Play Koi-Koi Hanafuda Card Games With Transformers](https://ieeexplore.ieee.org/document/10032777/), DOI 10.1109/TAI.2023.3240674. See RULES.md for the actual scoring contract and EXPERIMENTS.md for measured comparisons.

Beating this opponent would satisfy the named minimum baseline, not establish universal superiority over every Koi-Koi implementation. Preserve independent-model validation, untouched final deals and disclosure of development selection.

## Original-paper result interpretation

Inspected the repository PDF `TAI.2023.3240674.pdf`, SHA256 `5dc8d28f72bf5c1906003255f1c2f86e9c00999c4ab2d8ef3b25f15f4164a2fc`; extracted pages6-8 and visually checked complete page7/TableII. [Pinned source](https://github.com/guansanghai/KoiKoi-AI/blob/89d789565c7c36c84b74662e43001d960bf80901/TAI.2023.3240674.pdf).

TableII reports RL-WP versus RL-Point at51.5% over2,000 games, with point statistic-2.88. Human comparisons comprise only50 games per agent. The table gives the Author52% against RL-WP, while the prose gives RL-WP47% against humans; retain this discrepancy rather than silently reconciling it. RL-Point's reported human result is53%, not RL-WP's.

The paper defines its point statistic as final points minus initial30. Our `evaluate.duel` reports final candidate-minus-opponent points. Because the total remains60, our statistic is twice the paper's scale. Divide our point means and interval endpoints by2 when presenting the paper's convention; do not relabel historical reports or mix scales.

Decision: retain assisted RL-WP as the primary match-win comparator. Once a method is fixed, include assisted RL-Point as a secondary final comparator. Beating one policy does not imply beating the other, and neither establishes human superiority. This interpretation does not change the ongoing fixed endpoint.

## Open Hanafuda AI

Pinned snapshot: `suzukiplan/open-hanafuda-ai` at `01b4e484ac3d223f4ae8198d2a399f69078174d2`; remote HEAD and local checkout agree. Its [README](https://github.com/suzukiplan/open-hanafuda-ai/blob/01b4e484ac3d223f4ae8198d2a399f69078174d2/README.md) describes simulation, observable-information inference and heuristics, with MIT-licensed code.

The documented rules differ materially: one koi declaration per round, seven-point and opponent-koi doubling, initial-hand automatic outcomes, and a scoreless exhausted-hand result with dealer switching. Our paper-derived profiles have different declaration scoring and exhausted-hand outcomes. The README's claim of Nintendo conformity has not been independently audited here. Its example Hard-versus-Normal result is an internal comparison, not a comparison with assisted RL-WP.

Disposition: useful algorithmic reference and potential additional comparator, but not a drop-in opponent. A faithful port must explicitly reconcile scoring, legal actions, initial deals, card IDs and hidden-information boundaries. Changing its rules creates an adapted opponent and must be labelled accordingly. Do not infer its strength against our agents from its internal benchmark.

## Koikoi Swift

Pinned snapshot: `ngs/koikoi-swift` at `c8de6e1207ca4cdaf29d52cdd631fc26d4583538`; remote HEAD and local checkout agree. Inspected the actual [ISMCTS source](https://github.com/ngs/koikoi-swift/blob/c8de6e1207ca4cdaf29d52cdd631fc26d4583538/Sources/AI/ISMCTS.swift), rather than relying on the application description.

It implements single-observer determinization, availability-based UCB selection, heuristic rollouts and selection by root visit count. Search terminates at a round outcome; reward scales and clips round points. The configuration defaults to 800 iterations; the UI instantiates 400. These are different from our complete-match rollout objective and particle budget.

[Game.calcFinalScore](https://github.com/ngs/koikoi-swift/blob/c8de6e1207ca4cdaf29d52cdd631fc26d4583538/Sources/Core/Game.swift) applies seven-point and opponent-koi doubling, already sufficient to establish incompatibility with paper8. This was source inspection, not a build, exhaustive rules audit or strength evaluation. Source code is MIT-licensed; artwork is excluded under its [license](https://github.com/ngs/koikoi-swift/blob/c8de6e1207ca4cdaf29d52cdd631fc26d4583538/LICENSE).

Disposition: a reference for a separately specified tree-search experiment. Do not replace the current fixed full-action endpoint or call an adapted round-utility search a reproduction of its original playing strength.

## Experimental consequence

The earlier full-action search and confirmation studies did not replicate their win improvements; preserve those negative results. The completed cold/warm pretrained-PPO study independently qualifies both warm model seeds on development deals. Its fixed rule selected the warm method and first-seed canonical checkpoint.

The frozen final study is now complete and independently audited. Both selected models faced assisted RL-WP (primary) and assisted RL-Point (secondary), each on the same10,000 duplicate deals:80,000 played matches on10,000 unique deal streams. Both models clear the primary95% win-score lower-bound threshold above50%. The separate Point criterion does not replicate: canonical seed6710001's interval includes50%, and both models have negative expected point differences against RL-Point. There was no reranking, extension or pooled independence claim. The final deal range is consumed.

See STATUS.md and the manuscript for exact results, immutable protocol/artifact hashes and latency scope. Any future comparator still needs a pinned implementation, explicit rule compatibility, observation-only adapter checks and paired deals. Cross-variant results remain separate; computation scale alone is not strength evidence.
