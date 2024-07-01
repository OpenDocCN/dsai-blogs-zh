<!--yml
category: 未分类
date: 2024-07-01 18:18:29
-->

# Hacking git-rerere : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/01/hacking-git-rerere/](http://blog.ezyang.com/2010/01/hacking-git-rerere/)

An unusual workflow for Git, one that [Wizard](http://scripts.mit.edu/wizard) employs extensively, is when a single developer needs to perform merges inside lots of working copies. Normally, a maintainer would pull from the branches he cared about, and offload a large amount of the work to those who were interested in contributing patches. However, Wizard is using Git to provide a service for people who don't know and aren't interested in learning Git, so we need to push updates and merge their software for them.

The problem encountered when you are doing a lot of merging is "repeated resolution of the same conflicts." The solution, at least for the classical case, is `git rerere`. This feature saves the resolution of conflicts and then automatically applies those resolutions if the conflict comes up again. You can find out this much if you check `man git-rerere`.

Unfortunately, this merge resolution data is stored per `.git` directory, specifically in the `rr-cache` subdirectory, so some modest cleverness is necessary to make this work across many repositories. Fortunately, the simple solution of symlinking all of the `rr-cache` directories to a common directory both works *and* is safe of race-conditions when initially merging (it's not race-safe when writing out resolutions, but I am willing to consider this low contention enough to be a non-issue).

Why is this solution race safe? At first glance at the code in `rerere.c`, this would seem not to be the case: if two merges were to happen to generate the same merge conflict (precisely the use case of git rerere), the following code would get executed with the same value of `hex`:

```
ret = handle_file(path, sha1, NULL);
if (ret < 1)
        continue;
hex = xstrdup(sha1_to_hex(sha1));
string_list_insert(path, rr)->util = hex;
if (mkdir(git_path("rr-cache/%s", hex), 0755))
        continue;
handle_file(path, NULL, rerere_path(hex, "preimage"));

```

The last three lines access the (now shared) `rr-cache` directory, and `handle_file` will attempt to write out the file `rr-cache/$HEX/preimage` with the preimage contents; if both instances run `handle_file` concurrently, this file will get clobbered.

But, as it turns out, we don't care; barring a SHA-1 collision, both instances will write out the same file. The signature of `handle_file` is:

```
static int handle_file(const char *path,
         unsigned char *sha1, const char *output)

```

The first argument is a path to read conflict markers from, and is mandatory. `sha1` and `output` are optional; if `output` is not NULL, it contains a contents that the entire file, *minus* any diff3 conflict sections (the ones separated by `|||||||` and `=======`) gets written to; if `sha1` is not NULL, it gets a 20-byte binary digest written to it of the contents that `output` would have received. And thus, balance is restored to the world.

*Addendum*

Anders brings up the interesting question on whether or not two processes writing the same contents to the same file is actually race safe. Indeed, there is a very similar situation involving two processes writing the same contents to the same file which is a classic example of race conditions:

```
((echo "a"; sleep 1; echo "b") & (echo "a"; sleep 1; echo "b")) > test

```

Under normal circumstances, the contents of test is:

```
a
a
b
b

```

But every once in a while, one of the processes loses the race, and you get:

```
a
a
b

```

due to a non-atomic combination of writing and updating the file offset.

However, the distinguishing characteristic between this example and Git's case is that, in this example, there is only one file descriptor. However, in Git's case, there are two file descriptors, since each process called `open` independently. A more analogous shell script would be:

```
((echo "a"; sleep 1; echo "b") > test & (echo "a"; sleep 1; echo "b") > test)

```

the contents of which are (as far as I can tell) invariably:

```
a
b

```

Now, POSIX actually doesn't say what happens if two writes to the same offset with the same contents occur. However, casual testing seems to indicate that Linux and ext3 are able to give a stronger guarantee that writing the same values won't cause random corruption (note that, if the contents of the file were different, either combination could be possible, and this is what you see in practice).