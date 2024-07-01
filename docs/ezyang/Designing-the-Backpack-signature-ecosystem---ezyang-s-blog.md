<!--yml
category: 未分类
date: 2024-07-01 18:17:03
-->

# Designing the Backpack signature ecosystem : ezyang’s blog

> 来源：[http://blog.ezyang.com/2017/03/designing-the-backpack-signature-ecosystem/](http://blog.ezyang.com/2017/03/designing-the-backpack-signature-ecosystem/)

Suppose you are a library writer interested in using Backpack. Backpack says that you can replace a direct dependency on a function, type or package with one or more *signatures*. You typecheck against a signature and your end user picks how they want to eventually implement the signature.

Sounds good right? But there's a dirty little secret: to get all of this goodness, you have to *write* a signature--you know, a type signature for each function and type that you want to use in your library. And we all know how much Haskellers [hate writing signatures](https://ghc.haskell.org/trac/ghc/ticket/1409). But Backpack has a solution to this: rather than repeatedly rewrite signatures for all your packages, a conscientious user can put a signature in a package for reuse in other packages.

For the longest time, I thought that this was "enough", and it would be a simple matter of sitting down and writing some tutorials for how to write a signature package. But as I sat down and started writing signature packages myself, I discovered that there was more than one way to set things up. In the post, I want to walk through two different possible designs for a collection of signature packages. They fall out of the following considerations:

*   How many signature packages for, e.g., `bytestring`, should there be? There could be exactly one, or perhaps a separate *package* for each API revision?
*   Should it be possible to post a new version of a signature package? Under what circumstances should this be allowed?
*   For developers of a library, a larger signature is more convenient, since it gives you more functionality to work with. For a client, however, a smaller signature is better, because it reduces the implementation burden. Should signature packages be setup to encourage big or small signatures by default?

### A signature package per release

Intuitively, every release of a package is also associated with a "signature" specifying what functions that release supports. One could conclude, then, that there should be a signature package per release, each package describing the interface of each version of the package in question. (Or, one could reasonably argue that GHC should be able to automatically infer the signature from a package. This is not so easy to do, for reasons beyond the scope of this post.)

However, we have to be careful how we perform releases of each of these signatures. One obvious but problematic thing to do is this: given `bytestring-0.10.8.1`, also release a `bytestring-sig-0.10.8.1`. The problem is that in today's Haskell ecosystem, it is strongly assumed that only *one* version of a package is ever selected. Thus, if I have one package that requires `bytestring-sig == 0.10.8.1`, and another package that requires `bytestring-sig == 0.10.8.2`, this will fail if we try to dependency solve for both packages at the same time. We could make this scheme work by teaching Cabal and Stack how to link against multiple versions of a signature package, but at the moment, it's not practical.

An easy way to work around the "multiple versions" problem is to literally create a new package for every version of bytestring. The syntax for package names is a bit irritating (alphanumeric characters plus hyphens only, and no bare numbers between a hyphen), but you could imagine releasing `bytestring-v1008`, `bytestring-v1009`, etc., one for each version of the API that is available. Once a signature package is released, it should never be updated, except perhaps to fix a mistranscription of a signature.

Under semantic versioning, packages which share the same major version are supposed to only add functionality, not take it away. Thus, these successive signature packages can also be built on one another: for example `bytestring-v1009` can be implemented by inheriting all of the functions from `bytestring-v1008`, and only adding the new functions that were added in 0.10.9.

### A signature package per major release series

There is something very horrible about the above scheme: we're going to have *a lot* of signature packages: one per version of a package! How awful would it be to have in the Hackage index `bytestring-v900`, `bytestring-v901`, `bytestring-v902`, `bytestring-v1000`, `bytestring-v1002`, `bytestring-v1004`, `bytestring-v1006` and `bytestring-v1008` as package choices? (With perhaps more if there exist patch releases that accidentally changed the API.) Thus, it is extremely tempting to try to find ways to reduce the number of signature packages we need to publish.

Here is one such scheme which requires a signature package only for major releases; e.g., for `bytestring`, we would only have `bytestring-v9` and `bytestring-v10`:

*   The latest version of `bytestring-v9` should correspond to the "biggest" API supported by the 0.9 series. Thus, `bytestring-v9`, every minor version release of `bytestring`, there is a new release of `bytestring-v9`: e.g., when `bytestring-0.9.1.0` is released, we release `bytestring-v9-1.0`. Each of the releases increases the functionality recorded in the signature, but is not permitted to make any other changes.
*   When depending on the signature package, we instead provide a version bound specifying the minimum functionality of the signature required to build our package; e.g., `bytestring-v9 >= 1.0`. (Upper bounds are not necessary, as it assumed that a signature package never breaks backwards compatibility.)

There is one major difficulty: suppose that two unrelated packages both specify a version bound on `bytestring-v9`. In this case, the ultimate version of the signature package we pick will be one that is compatible with both ranges; in practice, the *latest* version of the signature. This is bad for two reasons: first, it means that we'll always end up requiring the client to implement the full glory of `bytestring-v9`, even if we are compatible with an earlier version in the release series. Second, it means that whenever `bytestring-v9` is updated, we may bring more entities into scope: and if that introduces ambiguity, it will cause previously compiling code to stop compiling.

Fortunately, there is a solution for this problem: use *signature thinning* to reduce the required entities to precisely the set of entities you need. For example, suppose that `bytestring-v9-0.0` has the following signature:

```
signature Data.ByteString where
    data ByteString
    empty :: ByteString
    null :: ByteString -> Bool

```

As a user, we only needed `ByteString` and `empty`. Then we write in our local `ByteString` signature:

```
signature Data.ByteString (ByteString, empty) where

```

and now *no matter* what new functions get added to `bytestring-v9-0.0`, this signature will only ever require `ByteString` and `empty`. (Another way of thinking about signature thinning is that it is a way to *centralize* explicit import lists.) Notice that this scheme does *not* work if you don't have a separate package per major release series, since thinning can't save you from a backwards incompatible change to the types of one of the functions you depend on.

These signature thinning headers can be automatically computed; I've [written a tool (ghc-usage)](https://hackage.haskell.org/package/ghc-usage) which does precisely this. Indeed, signature thinning is useful even in the first design, where they can be used to reduce the requirements of a package; however, with a signature package per major release, they are *mandatory*; if you don't use them, your code might break.

### Conclusion

So, what design should we adopt? I think the first scheme (a signature package per release) is more theoretically pure, but I am very afraid of the "too many packages" problem. Additionally, I do think it's a good idea to thin signatures as much as possible (it's not good to ask for things you're not going to use!) which means the signature thinning requirement may not be so bad. Others I have talked to think the first scheme is just obviously the right thing to do.

Which scheme do you like better? Do you have your own proposal? I'd love to hear what you think. (Also, if you'd like to bikeshed the naming convention for signature packages, I'm also all ears.)

### Appendix

After publishing this post, the comments of several folks made me realize that I hadn't motivated *why* you would want to say something about the API of bytestring-0.10.8; don't you just want a signature of strings? So, to address this comment, I want to describe the line of reasoning that lead me down this path.

I started off with a simple goal: write a signature for strings that had the following properties:

1.  Be reasonably complete; i.e., contain all of the functions that someone who wanted to do "string" things might want, but
2.  Be reasonably universal; i.e., only support functions that would be supported by all the major string implementations (e.g., String, strict/lazy Text, strict/lazy Word8/Char8 ByteString and Foundation strings.)

It turned out that I needed to drop quite a number of functions to achieve universality; for example, transpose, foldl1, foldl1', mapAccumL/R, scanl, replicate, unfoldr, group, groupBy, inits, tails are not implemented in Foundation; foldr', foldr1', scanl1, scanr, scanr1, unfoldN, spanEnd, breakEnd, splitOn, isInfixOf are not implemented by the lazy types.

This got me thinking that I could provide bigger signatures, if I didn't require the signature to support *all* of the possible implementations; you might have a signature that lets you switch between only the *strict* variants of string types, or even a signature that just lets you swap between Word8 and Char8 ByteStrings.

But, of course, there are combinatorially many different ways one could put signatures together and it would be horrible to have to write (and name) a new signature package for each. So what is the *minimal* unit of signature that one could write? And there is an obvious answer in this case: the API of a specific module (say, `Data.ByteString`) in a specific version of the package. Enter the discussion above.

### Appendix 2

Above, I wrote:

> But, of course, there are combinatorially many different ways one could put signatures together and it would be horrible to have to write (and name) a new signature package for each. So what is the *minimal* unit of signature that one could write? And there is an obvious answer in this case: the API of a specific module (say, `Data.ByteString`) in a specific version of the package.

I think there is an alternative conclusion to draw from this: someone should write a signature containing every single possible function that all choices of modules could support, and then have end-users responsible for paring these signatures down to the actual sets they use. So, everyone is responsible for writing big export lists saying what they use, but you don't have to keep publishing new packages for different combinations of methods.

I'm pursuing this approach for now!