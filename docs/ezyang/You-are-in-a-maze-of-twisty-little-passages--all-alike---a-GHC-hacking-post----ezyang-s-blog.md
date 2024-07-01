<!--yml
category: 未分类
date: 2024-07-01 18:18:03
-->

# You are in a maze of twisty little passages, all alike… (a GHC hacking post) : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/11/another-ghc-hacking-post/](http://blog.ezyang.com/2010/11/another-ghc-hacking-post/)

About a month ago I decided that it would be cool if I could solve the bug [GHC's runtime never terminates unused worker threads](http://hackage.haskell.org/trac/ghc/ticket/4262). Well, I just got around to looking at it today, and after wandering aimlessly around the twisty maze that is the GHC RTS for an hour or so, I finally found a light at the end of a tunnel, in the form of a heart-warmingly simple patch. I’ve sent mail off to Simon Marlow to make sure the light isn’t actually a train, but it occurred to me that it would be interesting to look at my command history and blog about the process by which I came to the conclusion that line 464 of `Capability.c` was the correct place to add my change, since this sort of mental journey is not the one that is really ever recorded anywhere in any shape or form.

* * *

*Warmups before running the maze.* In a shifty shifty maze like GHC, you want to make sure the guided route (i.e. a clean build) is working before trying anything fancy. I use a [separate build tree from source tree](http://hackage.haskell.org/trac/ghc/wiki/Building/Using#Sourcetreesandbuildtrees), so getting everything up to date involves:

```
cd ghc-clean
./darcs-all get
./darcs-all pull -a
cd ../ghc-build
lndir ../ghc-clean
perl boot && ./configure && make
inplace/bin/ghc-stage2 --interactive

```

When this has been resolved in a satisfactory manner (a non-trivial task for platforms with Windows), the code hunting can begin.

* * *

*Grab your equipment.* What? You mean to say you’ve wandered into this maze and you don’t even know how to tell you’ve gotten to your destination? That’s no good... you’ll need a dousing rod of some sort... something to tell you when you’ve got it right.

In this particular case, the original bug reporter had written up a small, incomplete test script, so the first thing I did was [flesh it out](http://hackage.haskell.org/trac/ghc/ticket/4262#comment:3) into a script that required no human interaction. The benchmark for the new script was clear: `/proc/PID/task` should report a number substantially smaller than 200\. To see that the current implementation is broken:

```
ezyang@javelin:~/Dev/ghc-build/testsuite/tests/ghc-regress/ffi/should_run$ ./cleanupThreads
203

```

* * *

*Getting your bearings.* Ok, so what do we want? We want threads to die instead of hanging around. There are two ways to do this: have the thread commit seppuku when it realizes it isn’t wanted, or have some manager kill the thread as necessary. The later is generally considered poor form, since you want to make sure the threads aren’t doing anything critical that will get corrupted if they die. So seppuku it is. Here, now, there are two questions:

1.  When does the thread decide to go into a waiting pool? This is presumably where we’d want it to terminate itself instead.
2.  How would the thread decide whether or not it should hang around or bug out?

* * *

*Mapping out the land.* GHC has this little runtime flag called `-Ds`. It’s pretty useful: it dumps out a whole gaggle of debug information concerning threads, which is precisely what we’d like to look for. Our plan of action is to look at what the thread activity looks like in our test script, and identify the points at which threads should be dying instead of hanging around.

The very beginning of the log looks like this:

```
b75006d0: allocated 1 capabilities
b75006d0: new task (taskCount: 1)
b75006d0: returning; I want capability 0
b75006d0: resuming capability 0
b75006d0: starting new worker on capability 0
b75006d0: new worker task (taskCount: 2)
b75006d0: task exiting
b75006d0: new task (taskCount: 2)
b75006d0: returning; I want capability 0
b71ffb70: cap 0: schedule()
b71ffb70: giving up capability 0

```

Note the number `b75006d0`; that’s our main thread and it’s going to be quite a busy beaver. Here is the very first thread we spin off to make a foreign call, but it finishes fairly quickly and isn’t the foreign call we are looking for:

```
b75006d0: cap 0: created thread 1
b75006d0: cap 0: thread 1 appended to run queue
b75006d0: new bound thread (1)
b75006d0: cap 0: schedule()
b75006d0: cap 0: running thread 1 (ThreadRunGHC)
b75006d0: cap 0: thread 1 stopped (suspended while making a foreign call)
b75006d0: freeing capability 0
b75006d0: returning; I want capability 0
b75006d0: resuming capability 0
b75006d0: cap 0: running thread 1 (ThreadRunGHC)
b75006d0: cap 0: thread 1 stopped (suspended while making a foreign call)
b75006d0: freeing capability 0
b75006d0: returning; I want capability 0
b75006d0: resuming capability 0
b75006d0: cap 0: running thread 1 (ThreadRunGHC)
b75006d0: cap 0: created thread 2
b75006d0: cap 0: thread 2 appended to run queue
b75006d0: cap 0: thread 1 stopped (finished)

```

Not before long, we see a veritable avalanche of new threads being created and added to the run queue—these are our threads:

```
b75006d0: woken up on capability 0
b75006d0: resuming capability 0
b75006d0: cap 0: running thread 3 (ThreadRunGHC)
b75006d0: cap 0: created thread 4
b75006d0: cap 0: thread 4 appended to run queue
b75006d0: cap 0: created thread 5
b75006d0: cap 0: thread 5 appended to run queue
b75006d0: cap 0: created thread 6
b75006d0: cap 0: thread 6 appended to run queue
b75006d0: cap 0: created thread 7
b75006d0: cap 0: thread 7 appended to run queue
b75006d0: cap 0: created thread 8
b75006d0: cap 0: thread 8 appended to run queue
b75006d0: cap 0: created thread 9
b75006d0: cap 0: thread 9 appended to run queue
b75006d0: cap 0: created thread 10
b75006d0: cap 0: thread 10 appended to run queue
b75006d0: cap 0: created thread 11
b75006d0: cap 0: thread 11 appended to run queue
b75006d0: cap 0: created thread 12
b75006d0: cap 0: thread 12 appended to run queue
b75006d0: cap 0: created thread 13

```

The process continues until we’ve spawned them all:

```
54139b70: starting new worker on capability 0
54139b70: new worker task (taskCount: 201)
53938b70: cap 0: schedule()
53938b70: cap 0: running thread 202 (ThreadRunGHC)
53938b70: cap 0: thread 202 stopped (suspended while making a foreign call)
53938b70: starting new worker on capability 0
53938b70: new worker task (taskCount: 202)
53137b70: cap 0: schedule()
53137b70: cap 0: running thread 203 (ThreadRunGHC)
53137b70: cap 0: thread 203 stopped (suspended while making a foreign call)
53137b70: starting new worker on capability 0
53137b70: new worker task (taskCount: 203)
52936b70: cap 0: schedule()

```

And then, since there’s nothing to do (all of our threads are in FFI land), we go and run a major GC:

```
   52936b70: woken up on capability 0
   52936b70: resuming capability 0
   52936b70: deadlocked, forcing major GC...
   52936b70: cap 0: requesting parallel GC
   52936b70: ready_to_gc, grabbing GC threads
all threads:
threads on capability 0:
other threads:
       thread  203 @ 0xb72b5c00 is blocked on an external call (TSO_DIRTY)
       thread  202 @ 0xb72b5800 is blocked on an external call (TSO_DIRTY)
       thread  201 @ 0xb72b5400 is blocked on an external call (TSO_DIRTY)
       thread  200 @ 0xb72b5000 is blocked on an external call (TSO_DIRTY)
       thread  199 @ 0xb72b4c00 is blocked on an external call (TSO_DIRTY)
       thread  198 @ 0xb72b4800 is blocked on an external call (TSO_DIRTY)
       thread  197 @ 0xb72b4400 is blocked on an external call (TSO_DIRTY)
       thread  196 @ 0xb72b4000 is blocked on an external call (TSO_DIRTY)
       thread  195 @ 0xb72b3c00 is blocked on an external call (TSO_DIRTY)
       thread  194 @ 0xb72b3800 is blocked on an external call (TSO_DIRTY)
       thread  193 @ 0xb72b3400 is blocked on an external call (TSO_DIRTY)
       [snip (you get the idea)]

```

(I’ve always kind of wondered whether or not FFI calls should be considered deadlocked.)

Now the threads start coming back from FFI-land and idling:

```
b69feb70: cap 0: running thread 4 (ThreadRunGHC)
b69feb70: cap 0: waking up thread 3 on cap 0
b69feb70: cap 0: thread 3 appended to run queue
b69feb70: cap 0: thread 4 stopped (finished)
b69feb70: giving up capability 0
b69feb70: there are 2 spare workers
b69feb70: passing capability 0 to bound task 0xb75006d0
b61fdb70: returning; I want capability 0
b61fdb70: resuming capability 0
b61fdb70: cap 0: running thread 5 (ThreadRunGHC)
b59fcb70: returning; I want capability 0
b61fdb70: cap 0: thread 5 stopped (finished)
b61fdb70: giving up capability 0
b61fdb70: there are 3 spare workers
b61fdb70: passing capability 0 to worker 0xb59fcb70
b75006d0: woken up on capability 0
b75006d0: capability 0 is owned by another task
b51fbb70: returning; I want capability 0
b59fcb70: resuming capability 0
b59fcb70: cap 0: running thread 6 (ThreadRunGHC)
b59fcb70: cap 0: thread 6 stopped (finished)
b59fcb70: giving up capability 0
b49fab70: returning; I want capability 0
b59fcb70: there are 4 spare workers
b59fcb70: passing capability 0 to worker 0xb51fbb70
b51fbb70: resuming capability 0
b51fbb70: cap 0: running thread 7 (ThreadRunGHC)
b51fbb70: cap 0: thread 7 stopped (finished)
b51fbb70: giving up capability 0
b41f9b70: returning; I want capability 0
b51fbb70: there are 5 spare workers

```

I've actually cheated a little: the `there are X spare workers` debug statements I added myself. But this section is golden; we’re specifically interested in these lines:

```
b61fdb70: cap 0: thread 5 stopped (finished)
b61fdb70: giving up capability 0

```

The thread stops, but it doesn’t die, it just gives up the capability. These are two extremely good candidates for where the thread might alternately decide to kill itself.

* * *

*Placemarkers.* It’s time to bust out the trusty old `grep` and figure out where these debug messages are being emitted from. Unfortunately, `5` and `finished` are probably dynamically generated messages, so `stopped` is the only real identifier. Fortunately, that’s specific enough for me to find the right line in the RTS:

```
ezyang@javelin:~/Dev/ghc-clean/rts$ grep -R stopped .
./Capability.c:    // list of this Capability.  A worker can mark itself as stopped,
./Capability.c:    if (!isBoundTask(task) && !task->stopped) {
./RaiseAsync.c:     - all the other threads in the system are stopped (eg. during GC).
./RaiseAsync.c:    // if we got here, then we stopped at stop_here
./Task.c:        if (task->stopped) {
./Task.c:    task->stopped       = rtsFalse;
./Task.c:    task->stopped = rtsFalse;
./Task.c:    task->stopped = rtsTrue;
./Task.c:    task->stopped = rtsTrue;
./Task.c:   debugBelch("task %p is %s, ", taskId(task), task->stopped ? "stopped" : "alive");
./Task.c:   if (!task->stopped) {
./sm/GC.c:      // The other threads are now stopped.  We might recurse back to
./Schedule.c:                  "--<< thread %ld (%s) stopped: requesting a large block (size %ld)\n",
./Schedule.c:                  "--<< thread %ld (%s) stopped to switch evaluators",
./Schedule.c:       // stopped.  We need to stop all Haskell threads, including
./Trace.c:        debugBelch("cap %d: thread %lu stopped (%s)\n",   ### THIS IS THE ONE
./Task.h:    rtsBool    stopped;         // this task has stopped or exited Haskell
./Task.h:// Notify the task manager that a task has stopped.  This is used
./Task.h:// Put the task back on the free list, mark it stopped.  Used by
./Interpreter.c:                  // already stopped at just now
./Interpreter.c:            // record that this thread is not stopped at a breakpoint anymore
./win32/Ticker.c:            // it still hasn't stopped.

```

That line in `Trace.c` is actually in a generic debugging function `traceSchedEvent_stderr`, but fortunately there’s a big case statement on one of its arguments `tag`:

```
case EVENT_STOP_THREAD:     // (cap, thread, status)
    debugBelch("cap %d: thread %lu stopped (%s)\n",·
               cap->no, (lnat)tso->id, thread_stop_reasons[other]);
    break;

```

So `EVENT_STOP_THREAD` is a good next grep. And sure enough:

```
ezyang@javelin:~/Dev/ghc-clean/rts$ grep -R EVENT_STOP_THREAD .
./Trace.c:    case EVENT_STOP_THREAD:     // (cap, thread, status)
./eventlog/EventLog.c:  [EVENT_STOP_THREAD]         = "Stop thread",
./eventlog/EventLog.c:        case EVENT_STOP_THREAD:     // (cap, thread, status)
./eventlog/EventLog.c:    case EVENT_STOP_THREAD:     // (cap, thread, status)
./Trace.h:    HASKELLEVENT_STOP_THREAD(cap, tid, status)
./Trace.h:    traceSchedEvent(cap, EVENT_STOP_THREAD, tso, status);

```

It looks to be an inline function in `Trace.h`:

```
INLINE_HEADER void traceEventStopThread(Capability          *cap    STG_UNUSED,
                                        StgTSO              *tso    STG_UNUSED,
                                        StgThreadReturnCode  status STG_UNUSED)
{
    traceSchedEvent(cap, EVENT_STOP_THREAD, tso, status);
    dtraceStopThread((EventCapNo)cap->no, (EventThreadID)tso->id,
                     (EventThreadStatus)status);
}

```

Classy. So `traceEventStopThread` is the magic word, and sure enough:

```
ezyang@javelin:~/Dev/ghc-clean/rts$ grep -R traceEventStopThread .
./Schedule.c:    traceEventStopThread(cap, t, ret);
./Schedule.c:  traceEventStopThread(cap, tso, THREAD_SUSPENDED_FOREIGN_CALL);
./Trace.h:INLINE_HEADER void traceEventStopThread(Capability          *cap    STG_UNUSED,

```

There are two plausible sites in `Schedule.c`.

* * *

*Going digging.* We first have to pick which site to inspect more closely. Fortunately, we notice that the second trace event corresponds to suspending the thread before going into a safe FFI call; that's certainly not what we're looking at here. Furthermore, the first is in the scheduler, which makes a lot of sense. But there’s nothing obvious in this vicinity that you might associate with saving a worker task away due to lack of work.

What about that `giving up` capability message? Some more grepping reveals it to be in the `yieldCapability` function (like one might expect). If we then trace backwards calls to `yieldCapability`, we see it is invoked by `scheduleYield`, which is in turn called by the scheduler loop:

```
scheduleYield(&cap,task);

if (emptyRunQueue(cap)) continue; // look for work again

// Get a thread to run
t = popRunQueue(cap);

```

This is very, very interesting. It suggests that the capability itself will tell us whether or not the work to do, and that `yieldCapability` is a promising function to look further into:

```
debugTrace(DEBUG_sched, "giving up capability %d", cap->no);

// We must now release the capability and wait to be woken up
// again.
task->wakeup = rtsFalse;
releaseCapabilityAndQueueWorker(cap);

```

That last call looks intriguing:

```
static void
releaseCapabilityAndQueueWorker (Capability* cap USED_IF_THREADS)
{
    Task *task;

    ACQUIRE_LOCK(&cap->lock);

    task = cap->running_task;

    // If the current task is a worker, save it on the spare_workers
    // list of this Capability.  A worker can mark itself as stopped,
    // in which case it is not replaced on the spare_worker queue.
    // This happens when the system is shutting down (see
    // Schedule.c:workerStart()).
    if (!isBoundTask(task) && !task->stopped) {
       task->next = cap->spare_workers;
       cap->spare_workers = task;
    }
    // Bound tasks just float around attached to their TSOs.

    releaseCapability_(cap,rtsFalse);

    RELEASE_LOCK(&cap->lock);
}

```

We’ve found it!

* * *

*Checking the area.* The `spare_workers` queue looks like the queue in which worker threads without anything to do go to chill out. We should verify that this is the case:

```
int i;
Task *t;
for (i = 0, t = cap->spare_workers; t != NULL; t = t->next, i++) {}
debugTrace(DEUBG_sched, "there are %d spare workers", i);

```

Indeed, as we saw in the debug statements above, this was indeed the case: the number of spare workers kept increasing:

```
54139b70: there are 199 spare workers
54139b70: passing capability 0 to worker 0x53938b70
53938b70: resuming capability 0
53938b70: cap 0: running thread 202 (ThreadRunGHC)
53938b70: cap 0: thread 202 stopped (blocked)
    thread  202 @ 0xb727a400 is blocked on an MVar @ 0xb72388a8 (TSO_DIRTY)
53938b70: giving up capability 0
53938b70: there are 200 spare workers
53938b70: passing capability 0 to worker 0x53137b70
53137b70: resuming capability 0
53137b70: cap 0: running thread 203 (ThreadRunGHC)
53137b70: cap 0: thread 203 stopped (blocked)
    thread  203 @ 0xb727a000 is blocked on an MVar @ 0xb72388a8 (TSO_DIRTY)
53137b70: giving up capability 0
53137b70: there are 201 spare workers

```

* * *

*Writing up the solution.* So, the patch from here is simple, since we’ve found the correct location. We check if the queue of spare workers is at some number, and if it is, instead of saving ourselves to the queue we just cleanup and then kill ourselves:

```
for (i = 1; t != NULL && i < 6; t = t->next, i++) {}
if (i >= 6) {
        debugTrace(DEBUG_sched, "Lots of spare workers hanging around, terminating this thread");
        releaseCapability_(cap,rtsFalse);
        RELEASE_LOCK(&cap->lock);
        pthread_exit(NULL);
}

```

And then we test see that this indeed has worked:

```
ezyang@javelin:~/Dev/ghc-build/testsuite/tests/ghc-regress/ffi/should_run$ ./cleanupThreads
7

```

* * *

*Postscript.* There are some obvious deficiencies with this proof-of-concept. It’s not portable. We need to convince ourselves that this truly does all of the cleanup that the RTS expects a worker to do. Maybe our data representation could be more efficient (we certainly don’t need a linked list if the number of values we’ll be storing is fixed.) But these are questions best answered by someone who knows the RTS better, so at this point I [sent in the proof of concept](http://www.haskell.org/pipermail/glasgow-haskell-users/2010-November/019503.html) for further review. Fingers crossed!