<!--yml
category: 未分类
date: 2024-07-01 18:17:15
-->

# How to maintain a pristine copy of your configuration files : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/01/how-to-maintain-a-pristine-copy-of-your-configuration-files/](http://blog.ezyang.com/2014/01/how-to-maintain-a-pristine-copy-of-your-configuration-files/)

[etckeeper](http://joeyh.name/code/etckeeper/) is a pretty good tool for keeping your /etc under version control, but one thing that it won’t tell you is what the diff between your configuration and a pristine version of your configuration (if you installed the same packages on the system, but didn’t change any configuration). [People have wanted this](https://blueprints.launchpad.net/ubuntu/+spec/foundations-q-dpkg-pristine-conffiles), but I couldn’t find anything that actually did this. A month ago, I figured out a nice, easy way to achieve this under etckeeper with a Git repository. The idea is to maintain a pristine branch, and when an upgrade occurs, automatically apply the patch (automatically generated) to a pristine branch. This procedure works best on a fresh install, since I don’t have a good way of reconstructing history if you haven’t been tracking the pristine from the start.

Here’s how it goes:

1.  Install etckeeper. It is best if you are using etckeeper 1.10 or later, but if not, you should replace [30store-metadata](https://github.com/joeyh/etckeeper/blob/master/pre-commit.d/30store-metadata) with a copy from the latest version. This is important, because pre-1.10, the metadata store included files that were ignored, which means you’ll get lots of spurious conflicts.

2.  Initialize the Git repository using `etckeeper init` and make an initial commit `git commit`.

3.  Create a pristine branch: `git branch pristine` (but stay on the master branch)

4.  Modify the etckeeper configuration so that `VCS="git"`, `AVOID_DAILY_AUTOCOMMITS=1` and `AVOID_COMMIT_BEFORE_INSTALL=1`:

    ```
    diff --git a/etckeeper/etckeeper.conf b/etckeeper/etckeeper.conf
    index aedf20b..99b4e43 100644
    --- a/etckeeper/etckeeper.conf
    +++ b/etckeeper/etckeeper.conf
    @@ -1,7 +1,7 @@
     # The VCS to use.
     #VCS="hg"
    -#VCS="git"
    -VCS="bzr"
    +VCS="git"
    +#VCS="bzr"
     #VCS="darcs"

     # Options passed to git commit when run by etckeeper.
    @@ -18,7 +18,7 @@ DARCS_COMMIT_OPTIONS="-a"

     # Uncomment to avoid etckeeper committing existing changes
     # to /etc automatically once per day.
    -#AVOID_DAILY_AUTOCOMMITS=1
    +AVOID_DAILY_AUTOCOMMITS=1

     # Uncomment the following to avoid special file warning
     # (the option is enabled automatically by cronjob regardless).
    @@ -27,7 +27,7 @@ DARCS_COMMIT_OPTIONS="-a"
     # Uncomment to avoid etckeeper committing existing changes to
     # /etc before installation. It will cancel the installation,
     # so you can commit the changes by hand.
    -#AVOID_COMMIT_BEFORE_INSTALL=1
    +AVOID_COMMIT_BEFORE_INSTALL=1

     # The high-level package manager that's being used.
     # (apt, pacman-g2, yum, zypper etc)

    ```

5.  Apply [this patch to etckeeper/commit.d/50vcs-commit](http://web.mit.edu/~ezyang/Public/etckeeper-pristine.patch). This patch is responsible for keeping the pristine branch up-to-date (more explanation below).

6.  Create a `.gitattributes` file with contents `.etckeeper merge=union`. This makes merges on the metadata file use the union strategy, which reduces spurious conflicts dramatically:

    ```
    diff --git a/.gitattributes b/.gitattributes
    new file mode 100644
    index 0000000..b7a1f4d
    --- /dev/null
    +++ b/.gitattributes
    @@ -0,0 +1 @@
    +.etckeeper merge=union

    ```

7.  Commit these changes.

8.  Permit pushes to the checked out `/etc` by running `git config receive.denyCurrentBranch warn`

9.  All done! Try installing a package that has some configuration and then running `sudo gitk` in `/etc` to view the results. You can run a diff by running `sudo git diff pristine master`.

So, what’s going on under the hood? The big problem that blocked me from a setup like this in the past is that you would like the package manager to apply its changes into the pristine etc, so that you can merge in the changes yourself on the production version, but it’s not obvious how to convince dpkg that `/etc` lives somewhere else. Nor do you want to revert your system configuration to pristine version, apply the update, and then revert back: this is just asking for trouble. So the idea is to apply the (generated) patch as normal, but then *reapply* the patch (using a cherry-pick) to the pristine branch, and then rewrite history so the parent pointers are correct. All of this happens outside of `/etc`, so the production copy of the configuration files never gets touched.

Of course, sometimes the cherry-pick might fail. In that case, you’ll get an error like this:

```
Branch pristine set up to track remote branch pristine from origin.
Switched to a new branch 'pristine'
error: could not apply 4fed9ce... committing changes in /etc after apt run
hint: after resolving the conflicts, mark the corrected paths
hint: with 'git add <paths>' or 'git rm <paths>'
hint: and commit the result with 'git commit'
Failed to import changes to pristine
TMPREPO = /tmp/etckeeper-gitrepo.CUCpBEuVXg
TREEID = 8c2fbef8a8f3a4bcc4d66d996c5362c7ba8b17df
PARENTID = 94037457fa47eb130d8adfbb4d67a80232ddd214

```

Do not fret: all that has happened is that the `pristine` branch is not up-to-date. You can resolve this problem by looking at `$TMPREPO/etc`, where you will see some sort of merge conflict. Resolve the conflict and commit. Now you will need to manually complete the rest of the script, this can be done with:

```
git checkout master
git reset --hard HEAD~ # this is the commit we're discarding
git merge -s ours pristine
git push -f origin master
git push origin pristine

```

To make sure you did it right, go back to `/etc` and run `git status`: it should report the working directory as clean. Otherwise, there are discrepancies and you may not have done the merges correctly.

I’ve been testing this setup for a month now, and it has proceeded very smoothly (though I’ve never attempted to do a full release upgrade with this setup). Unfortunately, as I’ve said previously, I don’t have a method for constructing a pristine branch from scratch, if you have an existing system you’d like to apply this trick to. There’s nothing stopping you, though: you can always decide to start, in which case you will record just the diffs from the time you started recording pristine. Give it a spin!