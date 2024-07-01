<!--yml
category: 未分类
date: 2024-07-01 18:17:41
-->

# Diskless Paxos crash recovery : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/08/diskless-paxos-crash-recovery/](http://blog.ezyang.com/2011/08/diskless-paxos-crash-recovery/)

*This is an edited version of an email I sent last week. Unfortunately, it does require you to be familiar with the original Paxos correctness proof, so I haven’t even tried to expand it into something appropriate for a lay audience. The algorithm is probably too simple to be in the literature, except maybe informally mentioned—however, if it is wrong, I would love to know, since real code depends on it.*

I would like to describe an algorithm for [Paxos](http://en.wikipedia.org/wiki/Paxos_algorithm) crash-recovery that does not require persistent storage, by utilizing synchronized clocks and a lattice-based epoch numbering. The basic idea is to increase the ballot/proposal number to one for which it is impossible for the crashed node to have made any promises for it. Such an algorithm, as noted in [Paxos made Live](http://labs.google.com/papers/paxos_made_live.html), is useful in the case of disk corruption, where persistent storage is lost. (Unfortunately, the algorithm they describe in the paper for recovering from this situation is incorrect. The reason is left as an exercise for the reader.) It is inspired by [Renesse's](http://www.cs.cornell.edu/home/rvr/) remark about an "epoch-based system", and the epoch-based crash-recovery algorithm described in [JPaxos: State Machine Replication Based on the Paxos Protocol](http://infoscience.epfl.ch/record/167765). However, in correspondence with [Nuno](http://personnes.epfl.ch/nuno.santos), I discovered that proofs for the correctness of their algorithm had not been published, so I took it upon myself to convince myself of its correctness, and in the process discovered a simpler version. It may be the case that this algorithm is already in the community folklore, in which case all the better, since my primary interest is implementation.

First, let's extend proposal numbers from a single, namespaced value n to a tuple `(e, n)`, where `n` is a namespaced proposal number as before, and `e` is an epoch vector, with length equal to the number of nodes in the Paxos cluster, and the usual Cartesian product lattice structure imposed upon it.

Let's establish what behavior we'd like from a node during a crash:

**KNOWN-UNKNOWNS.** An acceptor knows a value `e*`, for which for all e where `e* ≤ e` (using lattice ordering), the acceptor knows if it has responded to prepare requests of form `(e, n)` (for all `n`).

That is to say, the acceptor knows what set of proposal numbers he is guaranteed not to have made any promises for.

How can we establish this invariant? We might write a value to persistent storage, and then incrementing it upon a crash; this behavior is then established by monotonicity. It turns out we have other convenient sources of monotonic numbers: synchronized clocks (which are useful for Paxos in other contexts) have this behavior. So instead of using a vector of integers, we use a vector of timestamps. Upon a crash, a process sets its epoch to be the zero vector, except for its own entry, which is set to his current timestamp.

In [Paxos made Simple](http://academic.research.microsoft.com/Publication/12945610/paxos-made-simple), Lamport presents the following invariant on the operation of acceptors:

**P1a.** An acceptor can accept proposal numbered `n` iff it has not responded to a prepare request greater than `n`.

We can modify this invariant to the following:

**P1b.** An acceptor can accept proposal numbered `(e, n)` iff `e* ≤ e` and it has not responded to a prepare request `(_, n')` with `n' > n`.

Notice that this invariant "strengthens" **P1a** in the sense that an acceptor accepts a proposal in strictly less cases (namely, it refuses proposals when `e* ≰ e`). Thus, safety is preserved, but progress is now suspect.

When establishing progress of Paxos, we require that there exist a stable leader, and that this leader eventually pick a proposal number that is "high enough". So the question is, can the leader eventually pick a proposal number that is "high enough"? Yes, define this number to be `(lub{e}, max{n} + 1)`. Does this epoch violate **KNOWN-UNKNOWNS**? No, as a zero vector with a single later timestamp for that node is always incomparable with any epoch the existing system may have converged upon.

Thus, the modifications to the Paxos algorithm are as follows:

*   Extend ballot numbers to include epoch numbers;
*   On initial startup, set `e*` to be the zero vector, with the current timestamp in this node's entry;
*   Additionally reject accept requests whose epoch numbers are not greater-than or equal to `e*`;
*   When selecting a new proposal number to propose, take the least upper bound of all epoch numbers.

An optimization is on non-crash start, initialize `e*` to be just the zero vector; this eliminates the need to establish an epoch in the first round of prepare requests. Cloning state from a snapshot is an orthogonal problem, and can be addressed using the same mechanisms that fix lagging replicas. We recommend also implementing the optimization in which a leader only send accept messages to a known good quorum, so a recovered node does not immediately force a view change.

I would be remiss if I did not mention some prior work in this area. In particular, in [Failure Detection and Consensus in the Crash-Recovery Model](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.5958), the authors present a remarkable algorithm that, without stable storage, can handle more than the majority of nodes failing simultaneously (under some conditions, which you can find in the paper). Unfortunately, their solution is dramatically more complicated than solution I have described above, and I do not know of any implementations of it. Additionally, an alternate mechanism for handling crashed nodes with no memory is a group membership mechanism. However, group membership is notoriously subtle to implement correctly.