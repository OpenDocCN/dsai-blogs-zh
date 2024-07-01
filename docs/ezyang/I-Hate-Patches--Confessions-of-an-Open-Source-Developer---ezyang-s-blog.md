<!--yml
category: 未分类
date: 2024-07-01 18:18:18
-->

# I Hate Patches: Confessions of an Open-Source Developer : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/i-hate-patches-confessions-of-an-open-source-developer/](http://blog.ezyang.com/2010/05/i-hate-patches-confessions-of-an-open-source-developer/)

## I Hate Patches:
Confessions of an Open-Source Developer

It is a truth universally acknowledged that if you *really* want a change put into an open source project, you submit a patch along with the bug report. Sure, you might complain that the *average* user doesn't have any programming experience and that it's *unreasonable* to expect them to learn some complex system and then figure out how to make the change they're looking for, but you're just not in the secret society of hacker enthusiasts who fix their own damn software and give back to the projects they use.

I hate patches. I feel *morally obligated* to review and patch them in, and it usually ends up taking more than just "a few cycles."

Not all patches are created equal. I group them into this hierarchy:

*   *Cosmetically deficient.* The developer doesn't know anything about submitting patches; perhaps they haven't even discovered version control yet. They're apt to sending the entire modified file along with their changes. Those who do use `diff -u` don't bother looking at the output of the patch; they submit patches that swap spaces with tabs, random whitespace changes and gratuitous cosmetic changes. Many developers simply reject these patches.
*   *Semantically deficient.* For some developers, the act of crafting a patch is taking a source file, trying a vaguely plausible change, seeing if the change had the desired effect, and if not, try something else. In the degenerate case, the patch is nonsense, and in no way correct. More frequently, the submitted patch fails to account for common edge-cases in the application, appropriate error handling or interactions with other parts of the system. Many developers will reply nicely to patches like this and ask for a hunk to be done another way.
*   *Engineering deficient.* The patch is well-written, it looks good and does the right things. But... they didn't add tests to test the new changes, they didn't fix old unit tests changed by the functionality difference and they didn't add documentation in the appropriate places for the fix. Many developers will buckle down and make the engineering extras for the patch. Some developers don't have such tests (cough Linux kernel cough). Even more rarely, some projects can afford to make the patch submitter add the tests; usually this only occurs in projects that are aimed towards a fairly literate programming end-user community.

The Git mailing list can and does expect excellent patch submissions from its community; it's a version control system, that's the point! A library written in PHP used primarily by developers who have never written a unit test or submitted a unified diff for upstream review has much less flexibility. Most patches I receive for HTML Purifier never make it past the cosmetic deficiencies. And worse, the developers simply don't have the time to interactively improve the patch to the end: if I reply with a patch review, they never manage to get their patch to the point where it's acceptable for submission without excessive tooth pulling. But I feel guilty that my software is wrong, and so when I get the patch, I go and clean it up, incorporate it in, rewrite half of it, add the tests and then ship the change.

So, in the end, the software is improved by the submission by the patch, even if it didn't save the maintainer any time. So yeah, I hate patches. Maybe I should stop being *grumpy* and go back to *improving* my open source projects.