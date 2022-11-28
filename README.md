# Expected-Pass-Completion-xP

This repo explains how to build an Expected Pass Completion (xP) model and a Passes Completed Above Expected (PAx) score with football event-data.
Here I use Wyscout data from the API (v3 format), but you can leverage Wyscout free available data.

Why football clubs need such a model ?
- Because passes are the most recurrent events of a football game, which is why passing ability must be evaluated
- However, the pass completion rate metric is heavily biased (for instance, a center back performs mostly shorts and easy passes, unlike wingers or strikers)
- Thus, they need a new metric that indicates the probability of completed a pass (xP)
- It enables to reflect the difficulty and the risk taken by the player, and to value each pass (PAx)
