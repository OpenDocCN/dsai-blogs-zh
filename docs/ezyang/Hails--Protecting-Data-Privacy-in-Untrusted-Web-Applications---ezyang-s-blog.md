<!--yml
category: 未分类
date: 2024-07-01 18:17:25
-->

# Hails: Protecting Data Privacy in Untrusted Web Applications : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/10/hails-protecting-data-privacy-in-untrusted-web-applications/](http://blog.ezyang.com/2012/10/hails-protecting-data-privacy-in-untrusted-web-applications/)

*This post is adapted from the talk which Deian Stefan gave for Hails at OSDI 2012.*

It is a truth universally acknowledged that any website (e.g. Facebook) is in want of a web platform (e.g. the Facebook API). Web platforms are *awesome*, because they allow third-party developers to build apps which operate on our personal data.

But web platforms are also *scary*. After all, they allow *third-party* developers to build apps which operate on our *personal* data. For all we know, they could be selling our email addresses to spamlords or snooping on our personal messages. With the ubiquity of third-party applications, it’s nearly trivial to steal personal data. Even if we assumed that all developers had our best interests at heart, we'd still have to worry about developers who don't understand (or care about) security.

When these third-party applications live on untrusted servers, there is nothing we can do: once the information is released, the third-party is free to do whatever they want. To mitigate this, platforms like Facebook employ a CYA (“Cover Your Ass”) approach:

The thesis of the Hails project is that we can do better. Here is how:

First, third-party apps must be hosted on a trusted runtime, so that we can enforce security policies in software. At minimum, this means we need a mechanism for running untrusted code and expose trusted APIs for things like database access. Hails uses [Safe Haskell](http://www.haskell.org/ghc/docs/latest/html/users_guide/safe-haskell.html) to implement and enforce such an API.

Next, we need a way of specifying security policies in our trusted runtime. Hails observes that most data models have ownership information built into the objects in question. So a policy can be represented as a function on a document to a set of labels of who can read and a set of labels of who can write. For example, the policy “only Jen’s friends may see her email addresses” is a function which takes the a document representing a user, and returns the “friends” field of the document as the set of valid readers. We call this the MP of an application, since it combines both a model and a policy, and we provide a DSL for specifying policies. Policies tend to be quite concise, and more importantly are centralized in one place, as opposed to many conditionals strewn throughout a codebase.

Finally, we need a way of enforcing these security policies, even when untrusted code is being run. Hails achieves this by implementing thread-level dynamic information flow control, taking advantage of Haskell’s programmable semicolon to track and enforce information flow. If a third-party application attempts to share some data with Bob, but the data is not labeled as readable by Bob, the runtime will raise an exception. This functionality is called [LIO (Labelled IO)](http://hackage.haskell.org/package/lio), and is built on top of Safe Haskell.

Third-party applications run on top of these three mechanisms, implementing the view and controller (VC) components of a web application. These components are completely untrusted: even if they have security vulnerabilities or are malicious, the runtime will prevent them from leaking private information. You don’t have to think about security at all! This makes our system a good choice even for implementing official VCs.

One of the example applications we developed was [GitStar](http://www.gitstar.com/), a website which hosts Git projects in much the same way as GitHub. The key difference is that almost all of the functionality in GitStar is implemented in third party apps, including project and user management, code viewing and the wiki. GitStar simply provides MPs (model-policy) for projects and users. The rest of the components are untrusted.

Current web platforms make users decide between functionality and privacy. Hails lets you have your cake and eat it too. Hails is mature enough to be used in a real system; check it out at [http://www.gitstar.com/scs/hails](http://www.gitstar.com/scs/hails) or just `cabal install hails`.