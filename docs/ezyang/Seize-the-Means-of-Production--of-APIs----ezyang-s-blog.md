<!--yml
category: 未分类
date: 2024-07-01 18:17:04
-->

# Seize the Means of Production (of APIs) : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/09/seize-the-means-of-production-of-apis/](http://blog.ezyang.com/2016/09/seize-the-means-of-production-of-apis/)

There's a shitty API and it's ruining your day. What do you do?

Without making a moral judgment, I want to remark that there is something very different about these two approaches. In Dropbox's case, Dropbox has no (direct) influence on what APIs Apple provides for its operating system. So it has no choice but to work *within the boundaries* of the existing API. (When Apple says jump, you say, "How high?") But in Adam's case, POSIX is implemented by an open source project Linux, and with some [good ideas](http://dune.scs.stanford.edu/), Adam could implement his new interface *on top* of Linux (avoiding the necessity of writing an operating system from scratch.)

APIs cross social boundaries: there is the proletariat that produces software using an API and the bourgeoisie that controls the API. When the Man(TM) is a big corporation, our only choices are to work around their shitty API or pay them sufficiently large amounts of money to fix their shitty APIs. But when the Man(TM) is an open source project, your choices change. True, you could work around their shitty API. Or you could **seize the means of production**, thereby enabling you to fix the shitty API.

What do I mean by seize the means of production? Indeed, what *are* the means of production? An open source API does not live in a vacuum; it is made useful by the software that provides the API, the developers who contribute their time and expertise to maintain this expertise, even the publicity platform which convinces people to use an API. To seize the means of production is to gain control of all of these aspects. If you can convince the establishment that you are a core contributor of a piece of open source software, you in turn gain the ability to fix the shitty APIs. If you are unwilling or unable to do so, you might still fork, vendor or rewrite the project, but this is not so much seizing the means of production as it is recreating it from scratch. Another possibility is to build the abstraction you need on top of the existing APIs (as Adam did), though you are always at risk of the original project not being sensitive to your needs.

Time and time again, I see people working with open source projects who refuse to seize the means of production. Instead, they are willing to write increasingly convoluted workarounds to solve problems, all to stay within the boundaries. You might say, "This is just Getting Things Done(TM)", but at some point you will have done more work working around the problem, than you would have spent just fixing the damn thing.

So stop it. Stop putting up with shitty APIs. Stop working within the boundaries. Seize the means of production.

**Counterpoint.**

1.  What is advocated for in this post is nothing less than infinite yak shaving; if you take the advice seriously you will proceed to never get anything done ever again.
2.  It may be true that in aggregate, the cost of working around a bad API exceeds the cost to fix it, but for any individual the cost is generally less. Thus, even if you could perfectly predict the cost of a workaround versus a proper fix, individual incentives would prevent the proper fix.
3.  Users (including developers) don't know anything about the software they use and are ill-equipped to design better APIs, even if they know where it hurts.
4.  Rarely can you unilaterally seize the means of production. In an ideal world, to become a core contributor, it would merely be sufficient to demonstrate sustained, useful contributions to a project. We all know the real world is more messy.