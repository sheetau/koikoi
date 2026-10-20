Process-isolated auxiliary study recovery
========================================

The original learner failed during seed2_none update165, after all three
seed1 conditions completed. Both original process sessions exited1. Their
directories, failure receipts, models and logs remain untouched.

The failure occurs in os.add_dll_directory, before collection29 creates an
environment. CUDA backend construction registers the same 97-character Torch
DLL directory repeatedly without closing previous registrations. An isolated
probe reproduced WinError206 at332 successful registrations; explicitly closing
them restored successful registration. The probe performs no simulation or
learning. Its receipt is runs/score_auxiliary_dll_failure_probe.json,
SHA256 5cae1b76e15e5589f0545e15c1361a4f4b46ed2b9225f51f014943cbc06ac559.

Recovery preserves the frozen learner and native source. Each remaining arm
runs in a fresh process and creates at most101 CUDA backends. The original
training schedule, random seeds, objectives and evaluation criterion do not
change. This contains the DLL registration accumulation; a general backend
lifetime fix remains separate work after bound experiments finish.

The new directory is runs/score_auxiliary_recovery. Completed seed1 artifacts
are copied byte-for-byte; only endpoint paths in their complete.json receipts
are relocated. The original receipt and every original artifact are hash-bound.
The partial seed2_none is rerun from its original initialization. Every saved
old checkpoint tensor, optimizer, RNG, auxiliary collection and logged numerical
metric must agree with the rerun. Git provenance and wall time may differ.
Then seed2_win and seed2_score each run once in their own process.

Recovery protocol b38aa097e1ce9a4f2989eab445025d7f4388fbf97d7dd0df8722860bddbe5294
binds683inputs. The copied numerical protocol retains its original digest,
9ba0bd7b1d187d59269af336537ceab890bf8884a969a9df073dcd626e686f46.
All51,118,080 retained PPO decisions and76,800 requested auxiliary matches remain
the original six-arm comparison. Restart overhead is separate: the failed arm
executed165 PPO updates/2,703,360decisions, with164updates recorded in its log.
Logged work is2,686,976decisions/174,865round endings/22,736matches. Update165
round/match totals were not saved and must not be invented. Its28complete
auxiliary collections contain421,608decisions/27,488rounds/3,584matches. Repeated
training is not extra independent evidence.

Commands already launched; do not repeat into existing directories::

    .venv/Scripts/python.exe -m scripts.recover_score_auxiliary prepare
    .venv/Scripts/python.exe -m scripts.recover_score_auxiliary run
    .venv/Scripts/python.exe -m scripts.finish_score_auxiliary_recovery wait --producer-pid 67840

The recovery runner is session39422/PID67840; its first worker is67148.
Queue session56042/PID42884 holds the exact producer process, creation FILETIME
134369397674275076. Queue protocol
81de97227a5a280e31ecbc797bcaeb2e40f8fe189e51443f58191ac2213fc5c7 binds748inputs.
After producer exit0 and prefix equality, it runs the unchanged training auditor
against the recovery root, prepares the unchanged evaluation with that root,
runs all16fixed cells, then runs the unchanged evaluation auditor. The new
evaluation protocol additionally binds the relocation wrapper and recovery
evidence before any evaluation outcomes exist. The original failed queue is
preserved. No promotion or reserved101M use is performed.

Six recovery tests pass,0skips(1.67s); they check immutable copies, numerical
prefix agreement and rejection of altered metrics, RNG, optimizer fixtures and
collections. Two queue-adaptation tests pass,0skips(1.75s); the numerical design
is preserved and altered bindings are rejected. These tests use synthetic
files/tensors and perform no training, inference or simulation. Existing process
ownership/waiting and evaluation capability tests are reused without rerunning
their closed artifacts.

After the full training audit, mechanism commands must set STUDY to
Path("runs/score_auxiliary_recovery") in both score_auxiliary_mechanism and
audit_score_auxiliary_mechanism before calling their existing functions. Bind
the recovery protocol in the mechanism protocol before scoring. After the queue
completes, export_score_auxiliary_results must use the recovery TRAINING root
and the exact new QUEUE_SHA above, with the actual completed evaluation audit
digest. These are path/provenance adaptations; numerical diagnostics and all
six primary/120secondary coordinates remain fixed. Neither mechanism nor main
result export has run yet. Integrate actual results into the manuscript only
after these gates pass.
