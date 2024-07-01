<!--yml
category: 未分类
date: 2024-07-01 18:17:23
-->

# The GHC scheduler : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/01/the-ghc-scheduler/](http://blog.ezyang.com/2013/01/the-ghc-scheduler/)

I’d like to talk about some nitty-gritty details of GHC’s thread scheduling, discovered over the course of working on stride scheduling for GHC. Most of these choices are merely *implementation* details and are not part of any specification. While these choices shouldn’t be relied upon, they are worth knowing, since many of these details were accreted over the course of many performance bugs, benchmark tests and other battles. In this post, I’ll attempt to give some historical insight into why many choices were made. These insights should generalize to any system that would like to implement *green threads*, lightweight threads that use less memory than traditional operating system threads. For space reasons, I’m not going to talk about STM or sparks (though they are also quite interesting).

### Anatomy of a thread

I’d first like to discuss some brief background about the runtime system first and point out some perhaps nonintuitive design choices. A thread is represented by a TSO (thread-state object) by GHC, i.e. the `StgTSO` struct in `includes/rts/storage/TSO.h`. [1] In Haskell, TSOs can be passed around as `ThreadId` objects. The `Stg` in front of the struct name indicates that TSOs are *garbage collected*, like other closures in Haskell. The TSO, along with the stack allocated with it (STACK), constitute the primary memory overhead of a thread. Default stack size, in particular, is controlled by the GC flag `-ki`, and is 1k by default. [2] Threads are run by Capabilities, which can be thought of virtual cores managed by GHC. Capabilities are, in turn, mapped to true operating system threads, or Tasks, though we won’t talk about them much.

Being garbage collected has two major implications for TSOs. First, TSOs are *not* GC roots, so they will get GC'd if there is nothing holding on to them (e.g. [in the case of deadlock](http://blog.ezyang.com/2011/07/blockedindefinitelyonmvar/)), and their space is not *automatically* reclaimed when they finish executing [3]. Usually, a TSO will be retained by a Capability’s run queue (a GC root), or in the list of waiting threads of some concurrency variable, e.g. an MVar. Second, a TSO must be considered a *mutable* object, and is thus subject to the conventional GC write barriers necessary for any mutable object in a generational garbage collector. [4] The `dirty` bit tracks whether or not a TSO has been modified; it is always set when a thread is run and also when any of the pointer fields on a TSO are modified. Two fields, set by `setTSOLink` and `setTSOPrev`, are of particular interest to the scheduler.

### Run queue

The run queue is at the heart of the scheduler, as any runnable thread will hit the run queue before the scheduler actually pops it off the queue and runs it. There’s one per capability `rts/Capability.h` (in the bad old days, there was a global run queue, but this performed badly for multithreaded processes), and it is implemented as a doubly-linked list `run_queue_hd` and `run_queue_tl`. [6] The head and tail pointers mean that the queue is actually a deque: this is important because the scheduler will often have to handle threads that were interrupted in some way, and should let the threads get back on. The links themselves are on the TSOs and modified with `setTSOLink` and `setTSOPrev`, so modifying the queue dirties the TSOs involved. [7] Otherwise, the run queue is exclusively owned by the scheduler. If there are idle capabilities and if we have more than one thread left in our run queue, threads will be pushed to other queues with `schedulePushWork`.

Threads are put in *front* (`pushOnRunQueue`) if:

*   A stack overflow occurs;
*   A heap overflow occurs; [8]
*   A task attempts to run a thread, but it is [bound](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Concurrent.html#v:forkOS) and the current task is the wrong one;
*   A thread is associated with a black hole (a thunk that is being evaluated), and another thread, possibly on another capability, has blocked on its evaluation (see [ticket #3838](http://hackage.haskell.org/trac/ghc/ticket/3838));
*   In the threaded runtime, if a thread was interrupted because another Capability needed to do a stop-the-world GC (see commit `6d18141d8`);
*   In the non-threaded runtime, when a thread waiting on IO unblocks.

Threads are put in *back* (`appendToRunQueue`) in the case of pre-emption, or if it’s new; particularly, if

*   A thread was pre-empted via the context switch flag (e.g. incoming message from another thread, the timer fired, the thread cooperatively yielded, etc; see also [8] on how this interacts with heap overflows);
*   It is a new thread (so large amounts of thread creation do not starve old threads, see `conc004` and commit `05881ecab`);
*   A thread becomes unblocked;
*   A thread is migrated to another capability (though, in this case, the queue was empty anyway);
*   A thread finishes, but for some reason we need to keep it around (this is related to in-calls, though I’m not a 100% sure what is going on here; if you know, please tell me!)

### Conclusion

The GHC scheduler is pretty complicated! Much of the current behavior was created in response to specific problems: the right choices are *not* obvious a priori! I hope this post will serve as a valuable reference for any future GHC hackers interested in playing around with the scheduler, as well as for anyone else who needs to implement a scheduler for their runtime system. Much of the historical data was gleaned from comments (though I found some out-of-date ones), liberal use of `git blame`, and cross-referencing with the bug tracker—these are all useful places to figure out, “Well, why does that code do that?” In this post, I hope I’ve answered that question, to some degree.

* * *

[1] Initialization of `StgTSO` is handled in `createThread` in `rts/Threads.c`; this function is in turn invoked by `createGenThread`, `createIOThread` and `createStrictIOThread` in `rts/RtsAPI.c`. These functions setup the initial stack state, which controls what the thread executes when it actually gets run. These functions are the ones invoked by the `fork#` and other primops (entry-points for primops are located in `rts/PrimOps.cmm`).

[2] Actually, your usable stack will be a little smaller than that because this size also includes the size of the `StgTSO` struct. (This is only really for allocating lots of threads into one block, however, as once a GC occurs the TSOs and stacks will no longer be adjacent.)

[3] Here is a sample program which demonstrates how holding onto `ThreadId` using stable pointers (which force the object their pointing to to never be GC'd) can leak memory:

```
import Control.Concurrent
import Control.Monad
import Foreign.StablePtr

n = 400000
main = do
    ms <- replicateM n (newEmptyMVar >>= \m -> (forkIO (putMVar m ()) >>= newStablePtr) >> return m)
    mapM_ takeMVar ms

```

The [heap profile of the run](http://heap.ezyang.com/view/6e310e2e2e2c11ff3a7cc8ff0f5c205e51a8a188) shows none of the TSO/STACK objects being deallocated, even when the MVars drain out as threads finish executing.

[4] The write barrier for generational GCs refers not to [memory barrier](http://en.wikipedia.org/wiki/Memory_barrier) of multithreaded execution, but rather, notification for the garbage collector when a mutable reference in the old generation changes, and may now possibly point to an object in the young generation. Write barriers are necessary because the old generation will not be traversed during a minor collection, and thus if old generations may point to an object in a young generation, we may miss the fact that a young object is still alive even though it has no references from other young objects. In GHC, a write barrier is implemented by adding an object to the [mutable list](http://hackage.haskell.org/trac/ghc/wiki/StgObjectTypes) (`mut_list`) of a Capability if it is not in the youngest generation. (Some objects, like `MutArr#`, are *permanently* on the mutable list; in such a case, a write barrier may not be necessary. But see [5] for more details.) Objects will usually track their dirty status, so that they don’t add themselves to the mutable list multiple times. (Accidentally adding an object multiple times is harmless, but means the GC has to do extra work traversing the mutable list.) Additionally, if we can guarantee that the new reference does not point to the young generation (for instance, it is a static closure like `END_TSO_QUEUE`), then dirtying the object is not necessary. Getting this stuff right is tricky, to say the least!

[5] There is a bit of a sordid story here. Keeping an object permanently on the mutable list is done by `scavenge_mutable_list` in `rts/sm/Scav.c`, which will unconditionally re-add such an object to the mutable list if it sees it there. How does the object get on the mutable list in the first place? It’s not placed on the list upon creation; rather, upon the first minor GC on the youngest generation, the scavenging GC notices the object and places it on the mutable list by `gct->failed_to_evac = rtsTrue`. How do we end up freeing the object? The mutable list is considered a set of root pointers, but it is only *scavenged*, not evacuated. If an item on the mutable list ends up not being evacuated, it will be blown away regardless. (This does mean, however, that its elements will not be freed until the next GC.) Isn’t it really inefficient to always be scanning these arrays? Yes, and this used to [be a problem (ticket #650)](http://hackage.haskell.org/trac/ghc/ticket/650), nowadays mitigated by card marking. The same story applied to [TSOs (ticket #1589)](http://hackage.haskell.org/trac/ghc/ticket/1589), but the fix here was to properly apply a write barrier and not keep the objects permanently on the mutable list; this improved performance quite a bit when there were a lot of threads (even if you don’t scavenge their pointers, traversing a huge mutable list is still a pain.) Creating a lot of small mutable arrays is apt to be painful.

[6] It used to be singly linked, but fixing [ticket #3838](http://hackage.haskell.org/trac/ghc/ticket/3838) required the ability to remove TSOs from the run queue.

[7] Since these fields are always traversed by the GC, it’s important that they do not contain NULL pointers or garbage. Instead, we set them to the static closure `END_TSO_QUEUE`. Because this is guaranteed not to be in the young generation, this is why you do not need to dirty the TSO after setting this field.

[8] Sometimes, a heap overflow and a context switch occur simultaneously. If the thread requested a large block, we still always push it in front (because we don’t want another thread to steal our large block); however, otherwise, the context switch takes precedence and the thread is booted to the end of the queue—the context switch is checked as *late* as possible. (See commit `05881ecab`)