<!--yml
category: 未分类
date: 2024-07-01 18:18:09
-->

# Tension of patch and tree version control : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/09/tension-of-patch-and-tree-version-control/](http://blog.ezyang.com/2010/09/tension-of-patch-and-tree-version-control/)

*This post is not meant as a rag on Darcs, just a observation of the difference between two philosophies of version control. Also, I’m a bit new to Darcs, so there might be some factual inaccuracies. Please let me know about them!*

At some point, I would like to write a *Darcs for Git users* guide, distilling my experiences as an advanced Git user wrangling with Darcs. But perhaps the most important take away point is this: *don’t try to superimpose Git’s underlying storage model on Darcs!* Once I realized this point, I found Darcs fit rather nicely with my preferred Git development style—constant rebasing of local patches until they hit the official repository.

How does this rebasing workflow work? Despite the funny name, it’s a universal workflow that predates version control: the core operation is *submit a patch.* That is, after you're done hacking and recompiling and you've cleaned up your changes, you pull up the original copy of the repository, generate a unified diff, and send it off to the official mailing list. If the unified diff doesn’t apply cleanly to whatever the official development version is, upstream will ask you to make the patch apply to the newer version of the software.

Git streamlines this workflow with *rebases.* As the name suggests, you are changing the base commit that your patches are applied to. The identity of the patch is more important than the actual “history” of the repository. Interactive rebases allow you to reorder patches, and slice and dice history into something pretty for upstream to read.

Because Git supports both tree-based and patch-based workflows, the tension between the two schools of thought is fairly evident. Old commit objects become unreachable when you rebase, and you have to rely on mechanisms like the reflog to retrieve your old trees. Good practice is to never rebase published repositories, because once you’ve published a consistent history is more important than a pretty one.

Darcs *only* supports the patch-based workflow. It’s hard to keep your patches nicely ordered like you must do when you rebase, but there’s no need to: `darcs send --dry-run` will let you know what local patches that haven’t been put into the upstream repository are floating around, and essentially every interesting command asks you to explicitly specify what patch you are referring to with `-p`. Darcs makes it easy to merge and split patches, and edit old patches even if they’re deep within your `darcs changes` log.

However, there are times when I really do miss the tree-based model: in particular, while it’s easy to get close, there’s no easy way to get precisely the makeup of the repository as it was two days ago (when, say, your build was still working). The fact that Git explicitly reifies any given state your repository is in as a tree object makes the patch abstraction less fluid, but means you will *never ever lose committed data.* Unfortunately, with Darcs, there is no shorthand for “this particular repository state”; you might notice the patches that `darcs send` have to explicitly list *every* patch that came before the particular patch you’re sending out. In this way, I think Darcs is doing too much work: while the most recent N changes should be thought of as patches and not tree snapshots, I probably don’t care about the ancient history of the project. Darcs already supports this with tags, but my experience with fast moving repositories like GHC indicates to me that you also want a timeline of tags tracking the latest “official” repository HEAD.

There is also the subject of conflict resolution, but as I have not run into any of the complicated cases, I have little to say here.