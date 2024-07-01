<!--yml
category: 未分类
date: 2024-07-01 18:17:42
-->

# Synthetic Git merges : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/07/synthetic-git-merges/](http://blog.ezyang.com/2011/07/synthetic-git-merges/)

In theory, Git supports custom, low-level merge drivers with the `merge` configuration properties. In practice, no one actually wants to write their own merge driver from scratch. Well, for many cases where a custom merge driver would come in handy, you don’t have to write your merge driver from scratch! Consider these cases:

*   You want to merge files which have differing newline styles,
*   You want to merge files where one had lots of trailing whitespace removed,
*   You want to merge files one branch has replaced certain strings with custom strings (for example, a configuration file which instantiated `PASSWORD`, or a file that needs to be anonymized if there is a merge conflict),
*   You want to merge a binary file that has a stable textual format, or
*   You want to merge with knowledge about specific types of conflicts and how to resolve them (a super-smart `rerere`).

For all of these cases, you can instead perform a *synthetic Git merge* by modifying the input files (constructing synthetic merge inputs), calling Git’s `git merge-file` to do the actual merge, and then possibly editing the result, before handing it back off to the original invoker of your merge driver. It’s really simple. Here’s an example driver that handles files with differing newline styles by canonicalizing them to UNIX:

```
#!/bin/sh
CURRENT="$1"
ANCESTOR="$2"
OTHER="$3"
dos2unix "$CURRENT"
dos2unix "$ANCESTOR"
dos2unix "$OTHER"
exec git merge-file "$CURRENT" "$ANCESTOR" "$OTHER"

```

You can then set it up by frobbing your `.git/config`:

```
[merge.nl]
        name = Newline driver
        driver = /home/ezyang/merge-fixnewline.sh %A %O %B

```

And your `.git/info/attributes`:

```
*.txt merge=nl

```

In [Wizard](http://scripts.mit.edu/wizard/), we implemented (more clever) newline canonicalization, configuration value de-substitution (this reduces the diff between upstream and downstream, reducing the amount of conflicts due to proximity), and custom `rerere` behavior. I’ve also seen a coworker of mine use this technique manually to handle merge conflicts involving trailing whitespace (in Mercurial, no less!)

Actually, we took this concept further: rather than only create synthetic files, we create entirely synthetic trees, and then call `git merge` on them proper. This has several benefits:

*   We can now pick an arbitrary ancestor commit to perform the merge from (this, surprisingly enough, really comes in handy for our use-case),
*   Git has an easier time detecting when files moved and changed newline style, etc, and
*   It’s a bit easier to use, since you just call a custom command rather than have to remember how to setup your Git config and attributes properly (and keep them up to date!)

Merges are just metadata—multiple parents commits. Git doesn’t care how you get the contents of your merge commit. Happy merging!