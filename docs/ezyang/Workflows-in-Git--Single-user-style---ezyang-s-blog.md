<!--yml
category: 未分类
date: 2024-07-01 18:18:28
-->

# Workflows in Git: Single-user style : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/01/single-user-git-workflow/](http://blog.ezyang.com/2010/01/single-user-git-workflow/)

Nelson Elhage wrote a post about [Git and usability](http://blog.nelhage.com/archives/64), in which he discussed one of the reasons why Git seems to be so confusing to users who have come in straight from a Subversion-style workflow. When discussing this issue offline, one of the things that came up was the fact that, while Subversion imposes a fairly rigid workflow upon its users, Git is flexible enough to do almost any sort of workflow. This is terrible for a user placed in a shop that uses Git: when they go Google for how to use Git, they'll get any multitude of tutorials, each of which is for a *different workflow.*

In this multipart series, I'd like to discuss several different types of workflows that I've seen or experienced while using Git. This first post will look at a very simple example of a Git workflow, namely that of a single user, which will establish some basic idioms of Git that you might see in the other workflows.

A single-user workflow is, well, kind of simple. At it's simplest, it's not much more than a glorified backup system; you have lots of versions of your code. You can go back in time. Since I am assuming a general knowledge of version control systems, I don't think I need to convince you why this is useful. This article also assumes that you're comfortable enough to make commits in a repository (though we will *not* assume you know how to use the index; `-a` is a wondrous flag).

*Backups*

The very first thing you may notice when you move from a centralized VCS to a *decentralized* VCS is that your data never leaves your computer unless you explicitly say so. This is great if you are on an airplane and don't have Internet access; you don't have to pile up a stack of changes without being able to check in to the server. However, it means that you have to put in a little thought about where you are going to `push` your changes to. An easy way to do this is to utilize the multitude [free public hosting](http://git.or.cz/gitwiki/GitHosting). If you have a server that you have SSH access, private offsite backups are also easy: create a bare git repository on another server using `git init --bare` and then setup a remote that you can push to... but I'm getting ahead of myself!

If you created a Git repository and working copy on your own computer with `git init`, you'll now have to wrangle with Git remotes. I personally find this quite annoying, and thus always arrange to have my bare Git repository (i.e. the server) setup before I `git clone` my working copy (i.e. the client), which sets up the configuration that makes pushing easy. My steps are then:

# On my server, make a directory (I like `/srv/git/project.git`) and in it run `git init --bare` # On my client, run `git clone ssh://servername/srv/git/project.git`

If you must setup the remotes on an existing repository, the following commands will do the trick:

```
git remote add origin $REPO_URL
git config branch.master.remote origin
git config branch.master.merge refs/heads/master

```

For the curious, the first line adds a remote named "origin" (which, by convention, is the remote setup from the repository you may have cloned) associated with `$REPO_URL`. The second and third lines setup default behavior for when you pull changes from the repository, to simulate the configuration that normally gets setup when you do a clone. (Note: this kind of sucks. Git 1.7.0 introduces the `--set-upstream` flag which fixes these problems.)

From there, all you need to do is make commits with `git commit`, and then push them to the remote repository with `git push`.

*Topic branches*

As a single user, most of your work in your repository will play nicely together; you don't have to worry about someone else coming in and trampling on your commits. However, every once in a while you may find yourself in the midst of a large refactoring, and you find yourself having to leave things off for the day, or take an interrupt to work on a more pressing, albeit smaller, bugfix. Here, cheap commits and branching make this very simple on Git.

If you think the changes you are currently working on are big but you'll be able to get back immediately to them, use `git stash` to temporarily pop your changes into a stash. You can then perform your minor changes, and once done, use `git stash pop` to restore your old changes. Stash works best as a temporary scratch place for you to store changes, and should be immediately emptied out when possible; you don't want to be looking at multiple stashed changes and trying to figure out which one contains the ones you care about.

If your changes are a smidge bigger than that, or you think that you're not going to be able to work on whatever large change you're making for a while, you can make what's called a topic branch. First, change your working copy over to a new branch using `git checkout -b new-branch-name` (pick a descriptive name). Then, make a commit to save your changes. If you pop open `gitk`, you'll now notice that you have a commit hanging off of `master`. You can checkout master again using `git checkout master` and work on whatever other changes you need.

When you finally decide that your topic branch is done, you need to stick back into master. There are two ways to do this:

1.  You can pretend that your topic branch, as a whole, is just a big patch, and as such, this patch should reasonably apply to the most recent version of `master`. In that case, running `git rebase master` while on the topic branch (you can check with `git status`) will take this "patch" and apply it to master. You can then checkout master and `git pull topic-branch` to fast-forward master to the topic branch. Since getting rid of old branches is a good thing, I recommend running `git branch -d topic-branch` afterwards.
2.  You can take a stance that history is important, and perform a merge. On the master branch, run `git merge topic-branch`. Just as in the first case, you can then cleanup the topic branch with `git branch -d topic-branch`.

Cleaning up after old topic branches is a good habit to get into, because it means you can use `git branch` to remind yourself quickly which topic branches might need your attention.

Additionally, if you care about backing up your topic branches, you should run `git push origin topic-branch`. You can delete topic branches from your remote using `git push origin :topic-branch` (note the colon).

*Clean history*

Many people pay a lot of attention to documentation inside a source file in order to puzzle out what a particular piece of code does. However, another excellent source of code documentation is looking at the *history* of a piece of code; when did a particular snippet get introduced, and what explanation did the author give for it when making that change? `git blame` will give you a blow-by-blow description of when every particular line in a Git file was changed, and `git log` will show you the conglomeration of changes made to a particular file.

Unfortunately, the usefulness of this mechanism highly depends on the quality of the messages you're making in your commits, and if you're using Git properly and committing often, you might have skimped a little on some of the messages. No worries; it happens to the best of us. You just have to remember to *clean things up* (i.e. rewrite history) when you're done.

In this case, `git rebase -i` is your friend. Specify as an argument how far back you want to rewrite history (`HEAD~N` where N is a number is probably a good bet), and then rewrite history to your hearts content. You have three primary tools:

*   `edit`, and when Git gets to that commit, just run `git commit --amend`: This is fairly simple: you have a self-contained commit that you didn't really write a good commit message for, well amend will let you change that commit message into something that is useful.
*   `squash`: If you made a bunch of very small commits, and now you look at them and decide, no, they really logically go together, you can squash them together.
*   `edit` with `git checkout HEAD~`: What this will do is give you a working tree with the changes of that commit, but without any of them actually part of a commit. You can then break a "too big" commit into bite-sized pieces using `git add -p` (which will selectively add hunks of your changes to the index) and then using `git commit` *without* the `-a` flag).

This strategy interacts particularly well with topic branches, which lend themselves to the following workflow:

1.  Create the topic branch with `git checkout -b topic-name`,
2.  Hack a lot on the branch, making tiny commits with incomprehensible summaries,
3.  Review your changes with `git log -u master..HEAD`,
4.  Edit your changes with `git rebase -i master`,
5.  Checkout master and `git pull topic-name`.

And that's it for part one! You may have noticed that all of these strategies seem to feed into each other: this unusual integration between all aspects is one of the benefits of Git's simple internal model. If people would like to see some examples of these techniques in action, I'd be more than happy to blog about them some more. Thanks for reading.