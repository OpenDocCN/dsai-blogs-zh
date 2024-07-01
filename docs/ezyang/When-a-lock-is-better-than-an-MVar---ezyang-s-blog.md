<!--yml
category: 未分类
date: 2024-07-01 18:17:17
-->

# When a lock is better than an MVar : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/01/when-a-lock-is-better-than-an-mvar/](http://blog.ezyang.com/2014/01/when-a-lock-is-better-than-an-mvar/)

MVars are an amazingly flexible synchronization primitive, which can serve as locks, one-place channels, barriers, etc. or be used to form higher-level abstractions. As far as flexibility is concerned, MVars are the superior choice of primitive for the runtime system to implement—as opposed to just implementing, say, a lock.

However, I was recently thinking about [GHC's BlockedIndefinitelyOnMVar exception](http://blog.ezyang.com/2011/07/blockedindefinitelyonmvar/), and it occurred to me that a native implementation of locks could allow *perfect* deadlock detection, as opposed to the approximate detection for MVars we currently provide. (I must emphasize, however, that here, I define deadlock to mean a circular waits-for graph, and not “thread cannot progress further.”)

Here is how the new primitive would behave:

*   There would be a new type `Lock`, with only one function `withLock :: Lock -> IO a -> IO a`. (For brevity, we do not consider the generalization of Lock to also contain a value.)
*   At runtime, the lock is represented as two closure types, indicating locked and unlocked states. The locked closure contains a waiting queue, containing threads which are waiting for the lock.
*   When a thread takes out a free lock, it adds the lock to a (GC'd) held locks set associated with the thread. When it returns the lock, the lock is removed from this set.
*   When a thread attempts to take a busy lock, it blocks itself (waiting for a lock) and adds itself to the waiting queue of the locked closure.
*   Critically, references to the lock are treated as *weak pointers* when the closure is locked. (Only the pointer from the held lock set is strong.) Intuitively, just because a pointer to the lock doesn’t mean you can unlock; the only person who can unlock it is the thread who has the lock in their held locks set.
*   If a thread attempts to take out a lock on a dead weak pointer, it is deadlocked.

**Theorem.** *Any set of threads in a waits-for cycle is unreachable, if there are no other pointers to thread besides the pointer from the waiting queue of the locks in the cycle.*

**Proof.** Consider a single thread in the cycle: we show that the only (strong) pointer to it is from the previous thread in the cycle. When a thread is blocked, it is removed from the run queue (which counts as a GC root). Given the assumption, the only pointer to the thread is from the waiting queue of the lock it is blocked on. We now consider pointers to the lock it is blocked on. As this lock is busy, all pointers to it are weak, except for the pointer from the thread which is holding the lock. But this is exactly the previous thread in the cycle. ■

At the cost of a weak-pointer dereference when a lock is taken out, we can now achieve perfect deadlock detection. Deadlock will be detected as soon as a garbage collection runs that detects the dead cycle of threads. (At worst, this will be the next major GC.)

Why might this be of interest? After all, normally, it is difficult to recover from a deadlock, so while accurate deadlock reporting might be nice-to-have, it is by no means necessary. One clue comes from a sentence in Koskinen and Herlihy's paper [Dreadlocks: Efficient Deadlock Detection](http://www.cs.nyu.edu/~ejk/papers/dreadlocks-spaa08.pdf): “an application that is inherently capable of dealing with abortable lock requests...is software transactional memory (STM).” If you are in an STM transaction, deadlock is no problem at all; just rollback one transaction, breaking the cycle. Normally, one does not take out locks in ordinary use of STM, but this can occur when you are using a technique like transactional boosting (from the same authors; the relationship between the two papers is no coincidence!)

*Exercise for the reader, formulate a similar GC scheme for MVars restricted to be 1-place channels. (Hint: split the MVar into a write end and a read end.)*