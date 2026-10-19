Directional return diagnostic: complete results
==============================================

The planned continue-direction reversal is not established at either retained
parent. Every Monte Carlo mean and every estimator-minus-Monte-Carlo contrast
has a simultaneous95% interval containing zero. This closes the fixed diagnostic;
it neither establishes equivalence of estimators nor supports a reversal-based
explanation of the observed training plateau.

Collection and verification
---------------------------

The frozen CUDA policies complete8,192 sampled-self-play matches on4,096 shared
initial deal streams98110001..98114096. Actual work:952,157 decisions,64,067 round
endings and17,907 model-forward batches. Another193,891 terminal padded rows
consume sampling RNG but do not count as decisions. There are0 learning updates.
Both model parameter digests remain identical before and after collection.

The separate capability uses128 matches on64 shared initial deal streams,
15,163 decisions,1,013 round endings and280 model-forward batches. Its CPU replay
and scalar calculations pass before the main protocol is written. Capability
work is excluded from the8,192-match result.

The CPU auditor replays every recorded action and verifies initial/final states,
per-step state and public-input hashes,legal masks and work totals. Separate
forward TD expansions and scalar score calculations verify every advantage and
both-seat contribution. Maximum advantage discrepancy is8.881784197001252e-16.
Replay uses the shared transition core;it does not rerun neural inference or
action sampling. Replayed events are not additional independent matches.

Inference and interpretation
----------------------------

One70-coordinate family covers both parents,five public exponential-tilt probes,
four estimator means and three differences from MC.10,000 whole-deal bootstrap
replicates preserve the two parents and both seats. The fixed-SE maximum-deviation
critical value is3.255407047966088. All70 coordinates are nondegenerate. Bands
are asymptotic bootstrap intervals rather than finite-sample guarantees.

All10 MC-mean bands and all30 paired-difference bands include zero. Seven GAE
mean bands are positive:discard-preference full95/window95 at both parents,
discard-preference window1 at parent2,and koi-preference full95/window95 at
parent1. Those positive estimator means do not establish positive true gradients:
their MC comparisons remain inconclusive. Raising a base-probability preference
coefficient also preserves its greedy action ordering;it is not by itself a
new deployment action or evidence of improved complete-match play.

Units below are seat-averaged signed-return contributions per unit local
policy coefficient,not win percentages. The target is an average of two
unilateral responses against a frozen other seat. These are five selected
policy-family directions,not full neural gradients or gradients of joint
zero-sum payoff. Approximation and information effects are not disentangled.

All coordinates
---------------

.. csv-table:: Means and simultaneous95% intervals
   :header: "Parent", "Probe", "Estimator/contrast", "Mean", "Lower", "Upper"

   "parent1", "koi_continue", "mc", "-0.000535972", "-0.011899452", "0.010827509"
   "parent1", "koi_continue", "gae95_full", "0.001433692", "-0.002416289", "0.005283672"
   "parent1", "koi_continue", "gae95_window", "0.001598550", "-0.002128239", "0.005325338"
   "parent1", "koi_continue", "gae1_window", "-0.001597651", "-0.010560273", "0.007364971"
   "parent1", "koi_continue", "gae95_full-mc", "0.001969663", "-0.007440025", "0.011379352"
   "parent1", "koi_continue", "gae95_window-mc", "0.002134521", "-0.007623355", "0.011892398"
   "parent1", "koi_continue", "gae1_window-mc", "-0.001061679", "-0.008027055", "0.005903696"
   "parent1", "discard_preference", "mc", "0.029606234", "-0.001657544", "0.060870012"
   "parent1", "discard_preference", "gae95_full", "0.023579340", "0.013100658", "0.034058021"
   "parent1", "discard_preference", "gae95_window", "0.022287660", "0.012174207", "0.032401112"
   "parent1", "discard_preference", "gae1_window", "0.022118749", "-0.002327908", "0.046565406"
   "parent1", "discard_preference", "gae95_full-mc", "-0.006026894", "-0.032058705", "0.020004916"
   "parent1", "discard_preference", "gae95_window-mc", "-0.007318574", "-0.034433979", "0.019796831"
   "parent1", "discard_preference", "gae1_window-mc", "-0.007487485", "-0.028079068", "0.013104097"
   "parent1", "pick_hand_preference", "mc", "0.002091291", "-0.005464123", "0.009646704"
   "parent1", "pick_hand_preference", "gae95_full", "0.001665951", "-0.000680864", "0.004012766"
   "parent1", "pick_hand_preference", "gae95_window", "0.001896428", "-0.000337919", "0.004130775"
   "parent1", "pick_hand_preference", "gae1_window", "0.002781645", "-0.002992616", "0.008555905"
   "parent1", "pick_hand_preference", "gae95_full-mc", "-0.000425340", "-0.006739686", "0.005889006"
   "parent1", "pick_hand_preference", "gae95_window-mc", "-0.000194863", "-0.006734636", "0.006344911"
   "parent1", "pick_hand_preference", "gae1_window-mc", "0.000690354", "-0.003973459", "0.005354167"
   "parent1", "pick_draw_preference", "mc", "0.001572382", "-0.004260533", "0.007405297"
   "parent1", "pick_draw_preference", "gae95_full", "0.001144158", "-0.000804927", "0.003093243"
   "parent1", "pick_draw_preference", "gae95_window", "0.000920088", "-0.000953520", "0.002793696"
   "parent1", "pick_draw_preference", "gae1_window", "0.000547084", "-0.004113685", "0.005207853"
   "parent1", "pick_draw_preference", "gae95_full-mc", "-0.000428224", "-0.005260122", "0.004403673"
   "parent1", "pick_draw_preference", "gae95_window-mc", "-0.000652294", "-0.005649980", "0.004345393"
   "parent1", "pick_draw_preference", "gae1_window-mc", "-0.001025298", "-0.004649022", "0.002598426"
   "parent1", "koi_preference", "mc", "0.003795539", "-0.002695248", "0.010286326"
   "parent1", "koi_preference", "gae95_full", "0.002786144", "0.000514171", "0.005058117"
   "parent1", "koi_preference", "gae95_window", "0.002751187", "0.000533605", "0.004968769"
   "parent1", "koi_preference", "gae1_window", "0.003809813", "-0.001418336", "0.009037962"
   "parent1", "koi_preference", "gae95_full-mc", "-0.001009395", "-0.006372337", "0.004353547"
   "parent1", "koi_preference", "gae95_window-mc", "-0.001044352", "-0.006594547", "0.004505843"
   "parent1", "koi_preference", "gae1_window-mc", "0.000014274", "-0.003778814", "0.003807362"
   "parent2", "koi_continue", "mc", "0.003419206", "-0.007734951", "0.014573364"
   "parent2", "koi_continue", "gae95_full", "0.001069694", "-0.002581815", "0.004721204"
   "parent2", "koi_continue", "gae95_window", "0.000515204", "-0.003045775", "0.004076182"
   "parent2", "koi_continue", "gae1_window", "0.001640530", "-0.007044552", "0.010325613"
   "parent2", "koi_continue", "gae95_full-mc", "-0.002349512", "-0.011728972", "0.007029948"
   "parent2", "koi_continue", "gae95_window-mc", "-0.002904003", "-0.012660508", "0.006852503"
   "parent2", "koi_continue", "gae1_window-mc", "-0.001778676", "-0.009053837", "0.005496485"
   "parent2", "discard_preference", "mc", "0.028091631", "-0.004909081", "0.061092343"
   "parent2", "discard_preference", "gae95_full", "0.020424401", "0.009907055", "0.030941747"
   "parent2", "discard_preference", "gae95_window", "0.021459418", "0.011385926", "0.031532909"
   "parent2", "discard_preference", "gae1_window", "0.025321446", "0.000242121", "0.050400771"
   "parent2", "discard_preference", "gae95_full-mc", "-0.007667230", "-0.035321922", "0.019987462"
   "parent2", "discard_preference", "gae95_window-mc", "-0.006632213", "-0.035224373", "0.021959947"
   "parent2", "discard_preference", "gae1_window-mc", "-0.002770185", "-0.023639574", "0.018099205"
   "parent2", "pick_hand_preference", "mc", "-0.000843272", "-0.008284914", "0.006598370"
   "parent2", "pick_hand_preference", "gae95_full", "0.000543329", "-0.001771558", "0.002858216"
   "parent2", "pick_hand_preference", "gae95_window", "0.000504187", "-0.001735992", "0.002744366"
   "parent2", "pick_hand_preference", "gae1_window", "-0.000439858", "-0.006113552", "0.005233836"
   "parent2", "pick_hand_preference", "gae95_full-mc", "0.001386601", "-0.004923828", "0.007697030"
   "parent2", "pick_hand_preference", "gae95_window-mc", "0.001347459", "-0.005169596", "0.007864514"
   "parent2", "pick_hand_preference", "gae1_window-mc", "0.000403414", "-0.004366752", "0.005173581"
   "parent2", "pick_draw_preference", "mc", "0.001490050", "-0.004312990", "0.007293089"
   "parent2", "pick_draw_preference", "gae95_full", "0.001168191", "-0.000682590", "0.003018973"
   "parent2", "pick_draw_preference", "gae95_window", "0.001194577", "-0.000587044", "0.002976199"
   "parent2", "pick_draw_preference", "gae1_window", "0.001497636", "-0.002955181", "0.005950453"
   "parent2", "pick_draw_preference", "gae95_full-mc", "-0.000321859", "-0.005262552", "0.004618835"
   "parent2", "pick_draw_preference", "gae95_window-mc", "-0.000295472", "-0.005420539", "0.004829595"
   "parent2", "pick_draw_preference", "gae1_window-mc", "0.000007586", "-0.003887327", "0.003902499"
   "parent2", "koi_preference", "mc", "0.003148476", "-0.003241809", "0.009538761"
   "parent2", "koi_preference", "gae95_full", "0.001930508", "-0.000133713", "0.003994728"
   "parent2", "koi_preference", "gae95_window", "0.001861799", "-0.000123290", "0.003846888"
   "parent2", "koi_preference", "gae1_window", "0.001671864", "-0.003267333", "0.006611061"
   "parent2", "koi_preference", "gae95_full-mc", "-0.001217969", "-0.006647337", "0.004211400"
   "parent2", "koi_preference", "gae95_window-mc", "-0.001286677", "-0.006914033", "0.004340679"
   "parent2", "koi_preference", "gae1_window-mc", "-0.001476612", "-0.005487389", "0.002534165"

Evidence
--------

Commands::

   python -m pytest tests/test_directional_return_probe.py tests/test_directional_return_statistics.py -q
   python -m scripts.collect_directional_returns --mode capability --output runs/directional_return_capability
   python -m scripts.audit_directional_returns --input runs/directional_return_capability
   python -m scripts.collect_directional_returns --mode main --capability runs/directional_return_capability --output runs/directional_return_diagnostic
   python -m scripts.audit_directional_returns --input runs/directional_return_diagnostic

Five targeted tests pass,0 skips;Ruff passes. The new resampling test independently
expands sampled row frequencies and checks the common critical value and bands;
a separate fixture checks that parent/seat dependence remains within deal blocks.
The source/model/native-library protocol binds42 inputs. All main and capability
jobs are complete;do not restart or overwrite them.

SHA256 identities::

   runs/directional_return_capability/protocol.json
   afbc15d245a06c8a31c9c06ee4a15063674bd41a26cd9d061e664011dd3f8e09
   runs/directional_return_capability/report.json
   71d6308e8e6b1d24868e9649789e3ee0cec9a6c1b7f3d635821148c7b141aaa8
   runs/directional_return_capability/audit.json
   6cedec7c9afab1f0274cacb7d164ce6c04e54f39c1bf72ee2c114d3fb449713a
   runs/directional_return_diagnostic/protocol.json
   cae47d239af584ea62f47a97d879ecf83d7108870ca975b6a34db63a6c4ee911
   runs/directional_return_diagnostic/report.json
   5085458b15947062550d3a0732cb6a30d02bd9f29b988909c6085f50c4b7f059
   runs/directional_return_diagnostic/audit.json
   0aa47825725adf0ed1d7142edf1214c42c0653cfea555e1a72511104c608da3c
   runs/directional_return_diagnostic/summary.json
   5c63a84a8735520c73065e3a7e919f465ea5ddcea6b969794202430a6cf1c0c9

The stronger-play objective is still active. No model is promoted,and reserved
101M deals,dealer selection and the canonical player are unchanged. This result
does not justify another return-decay sweep. The next research decision must
identify a separate mechanism with measurable learning signal and distinguish
it from the already completed selector,return,public-memory and temperature
comparisons. The verified manuscript/PDF have not been revised by this study.
