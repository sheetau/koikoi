Longer pretrained-policy adaptation
==================================

Question and prior evidence
---------------------------

Does eight times as much policy optimization improve the retained pretrained
architecture beyond its500-update adaptation? The current records do not answer
this question. A scan of24 runs with kind=published config.json and metrics.jsonl
finds a largest within-run update count of520;this is a scoped inventory,not a
claim that no inherited lineage exceeds500 policy updates. The separate18-billion-
decision scale experiment trained an MLP under research12 rules and cannot answer
the question for the pretrained eight-round Transformer.

In retained seeds6710001/6720001,the last100 updates have mean approximate KL
0.0002343/0.0002374,gradient norm1.6617/1.6798 and entropy0.4887/0.4967. These
values show ongoing numerical updates,not convergence or expected improvement.
The completed directional diagnostic does not establish a score-weighted return
reversal. Temperature and return-estimator sweeps have already been tested.

Training specification
----------------------

Start two new learners from original published WP with seeds99110001/99210001.
Use the unchanged PublishedTrainer and PublishedConfig defaults:paper8,public
endgame assistance,CUDA float32,128 environments,128-step rollouts,512-row
minibatches,three PPO epochs,actorLR3e-6,valueLR1e-4,temperature0.05,lambda0.95,
entropy0.02 and256-wide public critic. Both receive20 critic-only warmup updates.
Save planned endpoints after500 and4,000 policy updates(total520 and4,020).
The short endpoint is the exact prefix of its long counterpart,including Adam,
environment and RNG state. Do not reset optimizer,change learning rate or select
an intermediate model. Additional snapshots support crash recovery only.

Each long run processes65,863,680 decisions including327,680 warmup decisions.
The short endpoint processes8,519,680 decisions. Incremental long-minus-short
work is57,344,000 decisions per seed. The two long runs together process
131,727,360 decisions;their short prefixes are included,not added again.
Round endings and completed matches are measured separately. The policy-update
budget ratio is8;total decisions including warmup have a different ratio.

This is an unequal-budget scaling comparison,not an equal-compute algorithm
comparison. Longer training itself is not claimed as a novel method. It tests
the strength component of the research objective while retaining the deployed
architecture and established return/representation analysis.

Capability and preservation
---------------------------

Use separate capability seeds99010001/99020001 with the production tensor shapes.
One critic-only update and two policy updates per seed test warmup stationarity,
actor movement and exact next-update restoration. The third unique update is
repeated after checkpoint restoration and must match metrics,model,Adam,
environment and RNG exactly. Count repeated verification work separately.
The shortened capability warmup is not the20-update main warmup.

Bind source,upstream source/weights,retained comparators,the completed directional
diagnostic and capability receipt before main training. Verify the original-WP
actor identity and zero output value layer at initialization. Main source or
input drift fails closed. Use new directories;never continue or overwrite a
closed historical study. Save initial,warmup,short,long and periodic recovery
checkpoints. A failed process remains failed until its saved state and metric
tail are reconciled;do not silently restart a learner.

Fixed development evaluation
-----------------------------

Evaluate all six policies:two short endpoints,two long endpoints and the two
retained parents. Each plays assisted WP and Point on4,096 duplicate deals per
opponent,strict singleton CPU inference,two threads,greedy,paper8 and public
endgame assistance. WP seeds99310001..99314096;Point99320001..99324096.
There are98,304 actual matches on8,192 shared initial deal streams. Both seats
and all policies share each opponent's deal set. Do not evaluate intermediate
snapshots or change training based on development outcomes. Evaluate only after
all planned training completes. Recorded-action CPU replay and independent
statistics/accounting checks precede any qualification.

The primary endpoint is draw-adjusted win score against the equal WP/Point
mixture. Compare long-minus-short and long-minus-retained-parent separately at
each training seed. Use one simultaneous95% family for these four contrasts.
For index i,average the WP and Point paired-deal contrasts at that index;resample
the4,096 compound blocks,retaining both opponents,all policies and both seats.
The two underlying deal streams per block remain distinct. Use10,000 bootstrap
replicates,seed99410001,fixed original standard errors,and the higher95th
percentile of the maximum absolute standardized centered mean deviation.
Zero-SE coordinates receive no positive-sign claim. This is an asymptotic
bootstrap procedure,not an exact finite-sample guarantee.

Qualification requires all four simultaneous lower bounds above zero. Report
all opponent-specific win and point results and their paired differences;
neither a secondary point improvement nor a favorable single seed replaces
the primary criterion. A mixture result does not establish superiority against
every opponent. If qualified,the canonical candidate is always seed99110001's
long endpoint,not the better-scoring seed. Reserved101M confirmation remains a
separate prerequisite for promotion. Otherwise retain the current player.

The trial has exactly two fresh training runs and the fixed evaluation above.
Do not extend failed endpoints,rerank snapshots or retune parameters within
this protocol. Preserve every result. The main manuscript will distinguish
any scaling gain from the separate research contribution;it will not rename
additional compute as algorithmic novelty.
