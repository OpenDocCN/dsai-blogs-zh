<!--yml
category: 未分类
date: 2024-07-01 18:18:09
-->

# pthread_cancel on Windows : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/09/pthread-cancel-on-window/](http://blog.ezyang.com/2010/09/pthread-cancel-on-window/)

> Edward, I’m afraid I have some bad news. Your [interruptible GHC patch](http://blog.ezyang.com/2010/08/interrupting-ghc/); it was involved in a terrible accident on the way to Windows portability. I hope you understand: we’re doing our best to patch it up, but there have been some complications...

Pop quiz! What does this pthreads code do?

```
#include <pthread.h>
#include <stdio.h>

void *thread1(void *arg) { sleep(10000); }
void *thread2(void *arg) { while (1) {} }

void *psycho_killer(void *arg) {
    pthread_t *id = (pthread_t*)arg;
    pthread_cancel(*id);
    printf("[%p] Psycho killer...\n", id);
    pthread_join(*id, NULL);
    printf("[%p] ...qu'est-ce que c'est.\n", id);
}

int main(char* argv, int argc) {
    pthread_t t1, t2, k1, k2;
    pthread_create(&t1, NULL, thread1, NULL);
    printf("[%p] I can't sleep 'cause my bed's on fire\n", &t1);
    pthread_create(&t2, NULL, thread2, NULL);
    printf("[%p] Don't touch me I'm a real live wire\n", &t2);
    pthread_create(&k1, NULL, psycho_killer, &t1);
    pthread_create(&k2, NULL, psycho_killer, &t2);
    pthread_join(k1, NULL);
    pthread_join(k2, NULL);
    printf("Run run run away!\n");
    return 0;
}

```

It never manages to terminate the second thread...

```
ezyang@javelin:~/Desktop$ ./test
[0xbf900b4c] I can't sleep 'cause my bed's on fire
[0xbf900b48] Don't touch me I'm a real live wire
[0xbf900b4c] Psycho killer...
[0xbf900b4c] ...qu'est-ce que c'est.
[0xbf900b48] Psycho killer...
^C

```

If you just had the `pthread_cancel` and the `pthread_setcancelstate` manpages, this might seem a little mysterious. The `pthreads` page, however, makes things clear: `sleep` is among one-hundred and two “cancellable” functions, which `pthread_cancel` must terminate within if a thread’s cancellability status is `PTHREAD_CANCEL_DEFERRED` (there are another two-hundred and forty-two which may or may not be cancelled). If the thread is stuck in userspace, it has to explicitly allow a deferred cancellation with `pthread_testcancel`. Previous versions of the POSIX spec were a little unclear whether or not cancellation should take place upon entry to the system call, or while the system call was running, but the [2008 spec](http://www.opengroup.org/onlinepubs/9699919799/functions/V2_chap02.html#tag_15_09_05_02) is fairly clear:

> Cancellation points shall occur when a thread is executing the following functions...

* * *

The million-dollar question is: “Can we implement the same semantics on Windows?” Actually, since it seems that a lot of people would have wanted pthreads functionality on Windows, you would think that this has been already been implemented by [pthreads-win32](http://sourceware.org/pthreads-win32/). We turn to the source!

```
if (tp->cancelType == PTHREAD_CANCEL_ASYNCHRONOUS
    && tp->cancelState == PTHREAD_CANCEL_ENABLE
    && tp->state < PThreadStateCanceling)
  {
    /* snip  */
  }
else
  {
    /*
     * Set for deferred cancellation.
     */
    if (tp->state < PThreadStateCancelPending)
      {
        tp->state = PThreadStateCancelPending;
        if (!SetEvent (tp->cancelEvent))
          {
            result = ESRCH;
          }
      }
    else if (tp->state >= PThreadStateCanceling)
      {
        result = ESRCH;
      }

    (void) pthread_mutex_unlock (&tp->cancelLock);
  }

```

Interestingly enough, pthreads-win32 doesn’t seem to do anything special: when we translate our test program and run it with pthreads-win32, it gets stuck on the `Sleep` call as well:

```
C:\Users\ezyang\pthreads-win32\Pre-built.2\lib>test.exe
[0022FF40] I can't sleep 'cause my bed's on fire
[0022FF38] Don't touch me I'm a real live wire
[0022FF40] Psycho killer...
[0022FF38] Psycho killer...
^C

```

* * *

At this point, it’s worth stepping back for a moment and asking, “What are we really trying to do here?” If you were to ask how to terminate threads on, say, Stack Overflow, you’d get a bunch of responses telling you, “Stop that and do it the right way”; namely, by explicitly handling thread termination on the thread itself via another message passing mechanism.

So there are number of different needs for interruptible calls:

1.  GHC would like to be able to put blocking IO calls on a worker thread but cancel them later; it can currently do this on Linux but not on Windows,
2.  Users would like to write interrupt friendly C libraries and have them integrate seamlessly with Haskell’s exception mechanism, and
3.  We’d like to have the golden touch of the IO world, instantly turning blocking IO code into nice, well-behaved non-blocking code.

Next time I’ll talk about what different approaches might be needed for each of these goals.