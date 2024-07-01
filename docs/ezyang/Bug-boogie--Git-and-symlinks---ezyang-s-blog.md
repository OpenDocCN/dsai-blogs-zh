<!--yml
category: 未分类
date: 2024-07-01 18:18:17
-->

# Bug boogie: Git and symlinks : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/06/bug-boogie-git-and-symlinks/](http://blog.ezyang.com/2010/06/bug-boogie-git-and-symlinks/)

Git is very careful about your files: unless you tell it to be explicitly destructive, it will refuse to write over files that it doesn't know about, instead giving an error like:

> Untracked working tree file 'foobar' would be overwritten by merge.

In my work with [Wizard](http://scripts.mit.edu/wizard), I frequently need to perform merges on working copies that have been, well, "less than well maintained", e.g. they untarred a new version of the the source tree on the old directory and forgot to add the newly added files. When Wizard goes in and tries to automatically upgrade them to the new version the proper way, this results in all sorts of untracked working tree file complaints, and then we have to go and manually check on the untracked files and remove them once they're fine.

There is a simple workaround for this: while we don't want to add all untracked files to the Git repository, we could add just the files that would be clobbered. Git will then stop complaining about the files, and we will still have records of them in the history:

```
def get_file_set(rev):
    return set(shell.eval("git", "ls-tree", "-r", "--name-only", rev).split("\n"))
old_files = get_file_set("HEAD")
new_files = get_file_set(self.version)
added_files = new_files - old_files
for f in added_files:
    if os.path.lexists(f): # *
        shell.call("git", "add", f)

```

Previously, the starred line of code read `if os.path.exists(f)`. Can you guess why this was buggy? Recall the difference betweeen `exists` and `lexists`; if the file in question is a symlink, `exists` will follow it, while `lexists` will not. So, if a file to be clobbered is a broken symlink, the old version of the code would not have removed it. In many case, you can't distinguish between the cases: if the parent directory of the file that the symlink points to exists, I can create a file via the symlink, and other normal "file operations."

However, Git is very keenly aware of the difference between a symlink and a file and will complain accordingly if it would have clobbered a symlink. Good ole information preservation!

*Postscript.* Yesterday was my first day of work at Galois! It was so exciting that I couldn't get my wits together to write a blog post about it. More to come.