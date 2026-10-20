Public final-round outcome bound coverage
========================================

This saved-data analysis covers all 28 completed source roots and all
33 available source root states. It performs no new transitions, model
calls, training updates, matches or independent strength measurements.
Keys are counted once within each root/method. Totals sum those counts;
they are not globally distinct states or probability-weighted coverage.
Fixed/optimized maps include every computed root action. Opponent caches
include evaluated alternatives. No cache of partial refused trees was
saved, so their interior coverage cannot be inferred here.

.. list-table:: Saved keys with certified outcomes, from the learner's perspective
   :header-rows: 1

   * - Kind
     - Keys
     - Win
     - Loss
     - Unresolved
     - Public contexts
   * - fixed
     - 305425
     - 25801
     - 0
     - 279624
     - 24325
   * - optimized
     - 299564
     - 25801
     - 0
     - 273763
     - 23529
   * - opponent
     - 552339
     - 11755
     - 1465
     - 539119
     - 29634

.. list-table:: Every available root
   :header-rows: 1

   * - Source
     - Completed
     - Final margin interval
     - Certificate
   * - 0
     - False
     - (-80, 56)
     - None
   * - 1
     - True
     - (-78, 60)
     - None
   * - 2
     - True
     - (10, 114)
     - 1
   * - 4
     - False
     - (-106, 12)
     - None
   * - 6
     - True
     - (-44, 74)
     - None
   * - 11
     - True
     - (-72, 70)
     - None
   * - 12
     - True
     - (-10, 90)
     - None
   * - 18
     - True
     - (-68, 86)
     - None
   * - 19
     - True
     - (-32, 80)
     - None
   * - 20
     - False
     - (-94, 42)
     - None
   * - 21
     - True
     - (-64, 54)
     - None
   * - 25
     - True
     - (-44, 76)
     - None
   * - 26
     - True
     - (-90, 186)
     - None
   * - 27
     - True
     - (-84, 18)
     - None
   * - 28
     - True
     - (-58, 68)
     - None
   * - 29
     - True
     - (-42, 84)
     - None
   * - 34
     - True
     - (4, 108)
     - 1
   * - 36
     - False
     - (-116, 12)
     - None
   * - 38
     - True
     - (-48, 70)
     - None
   * - 39
     - True
     - (-40, 78)
     - None
   * - 41
     - True
     - (-90, 38)
     - None
   * - 44
     - True
     - (-10, 90)
     - None
   * - 46
     - True
     - (-6, 112)
     - None
   * - 50
     - True
     - (-68, 86)
     - None
   * - 51
     - True
     - (-32, 80)
     - None
   * - 52
     - False
     - (-94, 42)
     - None
   * - 53
     - True
     - (-62, 56)
     - None
   * - 55
     - True
     - (-60, 86)
     - None
   * - 57
     - True
     - (-44, 76)
     - None
   * - 59
     - True
     - (-64, 24)
     - None
   * - 60
     - True
     - (-50, 76)
     - None
   * - 61
     - True
     - (-42, 84)
     - None
   * - 62
     - True
     - (-36, 92)
     - None

A certificate proves a fixed final win/loss under every legal continuation
of the local rules, independently of the opponent policy. Its absence
does not imply the outcome is changeable. Saved key coverage is not a
measured speedup or an estimate of physical transitions saved. An exact
planner still needs explicit certificate leaves, executable fallback
semantics, independent verification and a paired cost comparison.
