<!--yml
category: 未分类
date: 2024-07-01 18:17:06
-->

# Is no-reinstall Cabal coming to GHC 8.0? : ezyang’s blog

> 来源：[http://blog.ezyang.com/2015/09/is-no-reinstall-cabal-coming-to-ghc-8/](http://blog.ezyang.com/2015/09/is-no-reinstall-cabal-coming-to-ghc-8/)

You might be wondering: with the [beta release of no-reinstall Cabal](http://blog.ezyang.com/2015/08/help-us-beta-test-no-reinstall-cabal/), is this functionality be coming to GHC 8.0? (Or even a new release of Cabal, since the no-reinstall feature works with GHC 7.10). Unfortunately, there is a split among the Cabal developers over whether or not the actual no-reinstall behavior should go into Cabal by default as is. Duncan Coutts, in particular, has argued that it's a bad idea to enable no-reinstall without other (unimplemented) changes to Cabal. Since the extra needed changes are not fully implemented yet, it's unclear if Duncan will manage them for GHC 8.0.

I've heard a smattering of feedback that no-reinstall Cabal actually is working just fine for people, so I suspect many people would be in favor of just biting the bullet and putting in the "good" (but not "best") solution into Cabal. But I want to foster an informed discussion, so I'd like to explain what the (known) problems with no-reinstall are.

### What is no reinstall?

Currently, GHC and Cabal maintain an invariant in the installed package database that for any package name and version (i.e. (source) package ID), there is at most one matching package in the database:

The arrows indicate a "depends on" relationship: so if you have a database that has bar-0.1, bar-0.2 and an instance of foo-0.1 built against bar-0.1, you aren't allowed to install another instance of foo-0.1 built against bar-0.2 (though you are allowed to install foo-0.2 built against bar-0.2).

If cabal-install wants to install a package with the same package ID as a package already in the database, but with different dependencies, it must destructively overwrite the previous entry to maintain this invariant, pictured below:

No reinstall relaxes this invariant, so that "reinstalling" a package with different dependencies just works:

The recently beta released no-reinstall Cabal achieves this with two small features. First, in GHC 7.10, we added the flag `--enable-multi-instance` to `ghc-pkg` which makes `ghc-pkg` no longer error if you attempt to add multiple copies of the same package in the database. Second, in Vishal Agrawal's patchset for Cabal, cabal-install is modified to use this flag, so that the dependency solver no longer has to avoid reinstalls.

However, breaking this invariant has consequences. Let's look at some of them.

### Problem 1: It doesn't work on old versions of GHC

**Summary:** In GHC 7.8 and earlier, it's not possible to directly implement no reinstall (because `ghc-pkg` will reject it.) And even if it were possible, installing a new instance of a package (which has the same source package ID of an existing package) either (1) causes the old package and all of its dependents to become hidden from the default view of GHC, even though they are still usable, or (2) fails to be exposed in the default view of GHC.

Suppose that a package `foo-0.1`, which defines a type `Foo`, and has been compiled twice with different versions of its dependencies:

GHC 7.8 could not distinguish between two compilations of the package: symbols from both packages would live in the `foo-0.1` namespace, and colliding symbols would simply be considered the same. Disaster! To avoid this situation, GHC has a shadowing algorithm which remove incompatible packages from its visible set. Here is an example:

We have two package databases, the user database and the global database, laid side-to-side (the user database is "on top"). When there is a conflicting package ID in the combined database, GHC prefers the package from the topmost database: thus, in our example the global `foo-0.1` is shadowed (any packages which transitively have it as a dependency are also shadowed). When a package is shadowed, it doesn't exist at all to GHC: GHC will not suggest it or make any mention it exists.

No reinstall requires us to allow these duplicate packages the same database! In this case, GHC will apply shadowing; however, it is not well-defined which package should be shadowed. If GHC chooses to shadow the old package, they "vanish" from GHC's default view (it is as if they do not exist at all); if GHC chooses to shadow the new package, a package that a user just `cabal-install`'d will be mysteriously absent! Troublesome.

### Problem 2: Using multiple instances of the same package is confusing

**Summary:** In GHC 7.10 or later, multiple instances of the same package may be used together in the same GHC/GHCi session, which can result in confusing type inequalities.

In GHC 7.10, we now use "package keys" to test for type identity. A package key is a source package ID augmented with a hash of the package keys of all the dependencies. This means that GHC no longer needs to apply shadowing for soundness, and you can register duplicates of a package using the `--enable-mult-instances` flag on `ghc-pkg`.

However, this can still result in confusing behavior. Consider the previous example in GHC 7.10:

Both versions of `foo` are visible, and so if we try to import `Foo`, GHC will complain that it doesn't know which `Foo` we want. This can be fixed by hiding one package or the other. However, suppose that both `baz` and `qux` are exposed, and furthermore, they both export a value `foo` which has type `Foo`. These types are "distinct", despite the fact that they are: (1) both named `Foo`, and (2) come from a package named `foo-0.1`: they are two different instances of `foo-0.1`. Confusing!

### Problem 3: Nix hashing non-sdist'ed packages is difficult

It is easy to "trick" Cabal into hashing a set of source files which is not representative of the true input of the build system: for example, you can omit files in the `other-modules` field, or you can modify files in between the time Cabal has computed the source hash and the time it builds the files. And if you can't trust the Nix hash, you now have to worry about what happens when you really need to clobber an old entry in the Nix database (which incorrectly has the "same" hash as what you are attempting to install).

This problem doesn't exist for tarballs downloaded from Hackage, because you can simply hash the tarball and that is guaranteed to be the full set of source for building the file.

### Duncan's patchset

To deal with these problems, Duncan has been working on a bigger patchset, with the following properties:

1.  To support old versions of GHC, he is maintaining a separate "default view" package database (which is used by bare invocations of GHC and GHCi) from the actual "Nix store" package database. `cabal-install` is responsible for maintaining a consistent default view, but also installs everything into the Nix store database.
2.  Nix-style hashing is only done on Hackage packages; local source tree are to be built and installed only into a sandbox database, but never the global database. Thus, an actual Nix hash is only ever computed by `cabal-install`.
3.  He also wants to make it so that `cabal-install`'s install plan doesn't depend on the local state of the Nix database: it should give the same plan no matter what you have installed previously. This is done by dependency resolving without any reference to the Nix database, and then once IPIDs are calculated for each package, checking to see if they are already built. This plan would also make it possible to support `cabal install --enable-profiling` without having to blow away and rebuild your entire package database.

### Vishal's patchset

Vishal was also cognizant of the problems with the default view of the package database, and he worked on [some patches](https://phabricator.haskell.org/D1119) to GHC for support for modifying package environments, which would serve a similar role to Duncan's extra package databases. Unfortunately, these patches have been stuck in code review for a bit now, and they wouldn't help users of old versions of GHC. While the code review process for these patches may get unstuck in the near future, I'm hesitant to place any bets on these changes landing.

### Conclusion

My view is that, historically, problems one and two have been the big stated reasons why "no reinstall", while being a simple change, hasn't been added to Cabal as the default mode of operation. However, there's been rising sentiment (I think I can safely [cite Simon Marlow](https://www.reddit.com/r/haskell/comments/3ite8n/noreinstall_cabal_a_project_to_move_cabal_to_a/cuk7gn9) in this respect) among some that these problems are overstated, and that we should bite the bullet.

If we want to turn on "no reinstall" before Duncan finishes his patchset (whenever that will be—or maybe someone else will finish it), I think there will need to be some concerted effort to show that these problems are a small price to pay for no reinstall Cabal, and that the Haskell community is willing to pay... at least, until a better implementation comes around.