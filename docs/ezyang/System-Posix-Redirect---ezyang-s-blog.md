<!--yml
category: 未分类
date: 2024-07-01 18:18:13
-->

# System.Posix.Redirect : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/07/system-posix-redirect/](http://blog.ezyang.com/2010/07/system-posix-redirect/)

[System.Posix.Redirect](http://hackage.haskell.org/package/system-posix-redirect) is a Haskell implementation of a [well-known, clever and effective POSIX hack](http://homepage.ntlworld.com/jonathan.deboynepollard/FGA/redirecting-standard-io.html). It’s also completely fails software engineering standards. About a week ago, I excised this failed experiment from my work code and uploaded it to Hackage for strictly academic purposes.

*What does it do?* When you run a command in a shell script, you have the option of *redirecting* its output to another file or program:

```
$ echo "foo\n" > foo-file
$ cat foo-file
foo
$ cat foo-file | grep oo
foo

```

Many APIs for [creating new processes](http://www.haskell.org/ghc/docs/6.12.2/html/libraries/process-1.0.1.2/System-Process.html) which allow custom stdin/stdout/stderr handles exist; what System.Posix.Redirect lets you do is redirect stdout/stderr without having to create a new process:

```
redirectStdout $ print "foo"

```

*How does it do it?* On POSIX systems, it turns out, almost exactly the same thing that happens when you create a subprocess. We can get a hint by strace'ing a process that creates a subprocess with slightly different handles. Consider this simple Haskell program:

```
import System.Process
import System.IO

main = do
    -- redirect stdout to stderr
    h <- runProcess "/bin/echo" ["foobar"] Nothing Nothing Nothing (Just stderr) Nothing
    waitForProcess h

```

When we run `strace -f` (the `-f` flag to enable tracking of subprocesses), we see:

```
vfork(Process 26861 attached
)                                 = 26861
[pid 26860] rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
[pid 26860] ioctl(0, SNDCTL_TMR_TIMEBASE or TCGETS, {B38400 opost isig icanon echo ...}) = 0
[pid 26860] ioctl(1, SNDCTL_TMR_TIMEBASE or TCGETS, {B38400 opost isig icanon echo ...}) = 0
[pid 26860] waitpid(26861, Process 26860 suspended
 <unfinished ...>
[pid 26861] rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
[pid 26861] dup2(0, 0)                  = 0
[pid 26861] dup2(2, 1)                  = 1
[pid 26861] dup2(2, 2)                  = 2
[pid 26861] execve("/bin/echo", ["/bin/echo", "foobar"], [/* 53 vars */]) = 0

```

The `dup2` calls are the key, since there are no special arguments to be passed to `vfork` or `execve` (the “53 vars” are the inherited environment) to fiddle with the standard handles of the subprocess, we need to fix them ourself. `dup2` copies the file descriptor `2`, guaranteed to be stderr (`0` is stdin, and `1` is stdout), onto `stdout`, which is what we asked for in the original code. File descriptor tables are global to a process, so when we change the file descriptor `1`, everyone notices.

There is one complication when we are not planning on following up the `dup2` call with an `execve`: your standard library may be buffering output, in which case there might have been some data still living in your program that hasn’t been written to the file descriptor. If you play this trick in a normal POSIX C application, you only need to flush the `FILE` handles from `stdio.h`. If you’re in a Haskell application, you also need to flush Haskell’s buffers. (Notice that this isn’t necessary if you `execve`, since this system call blows away the memory space for the new program.)

*Why did I write it?* I had a very specific use-case in mind when I wrote this module: I had an external library written in C that wrote error conditions to standard output. Imagine a `hPutStr` that printed an error message if it wasn’t able to write the string, rather than raising an `IOException`; this would mean terrible things for client code that wanted to catch and handle the error condition. Temporarily redirecting standard output before calling these functions means that I can marshal these error conditions to Haskell while avoiding having to patch the external library or having to relegate it to a subprocess (which would cause much slower interprocess communication).

*Why should I not use it in production?* “It doesn’t work on Windows!” This is not 100% true: you could get a variant of this to work in some cases.

The primary problem is the prolific selection of runtimes and standard libraries available on Windows. Through some stroke of luck, the vast majority of applications written for Unix use a single standard library: libc, and you can be reasonably certain that you and your cohorts are using a single `FILE` abstraction, and since file descriptors are kernel-side, they’re guaranteed to work no matter what library you’re using. No such luxury on Windows: that DLL you’re linking against probably was compiled by some other compiler toolchain with it’s own runtime. GHC, in particular, uses the MingW toolchain to link on Windows, whereas native code is much more likely to have been compiled with Microsoft tools (MSVC++, anyone?). If the library could be recompiled with MingW, it could have worked, but I decided that it would be easier to just patch the library to return error codes another way. And so this module was obliterated from the codebase.