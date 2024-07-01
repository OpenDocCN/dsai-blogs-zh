<!--yml
category: 未分类
date: 2024-07-01 18:17:04
-->

# The Base of a String Theory for Haskell : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/09/the-base-of-a-string-theory-for-haskell/](http://blog.ezyang.com/2016/09/the-base-of-a-string-theory-for-haskell/)

One of the early posts from this blog, from 2010, was on the subject of [how to pick your string library in Haskell](http://blog.ezyang.com/2010/08/strings-in-haskell/). Half a decade later, the Haskell ecosystem is still largely in the same situation as it was half a decade ago, where most of the boot libraries shipped with GHC (e.g., `base`) still use the `String` type, despite the existence of superior string types. The problem is twofold:

1.  No one wants to break all of the existing code, which means libraries like `base` have to keep `String` versions of all their code. You can't just search-replace every occurrence of `String` with `Text`.
2.  No one wants to be in the business of maintaining *two* copies of any piece of code, which are copy-pastes of each other but subtly different. In practice, we must: e.g., [unix](https://hackage.haskell.org/package/unix) has `ByteString` variants of all of its functions (done by copy-paste); [text](https://hackage.haskell.org/package/text) provides some core IO functionality (also done by copy-paste). But it is terrible and scales poorly: every downstream library that wants to support two string types (or more) now has to publish two copies of themselves, and any new string implementation has the unenviable task of reimplementing the world to make themselves useful.

Backpack solves these problems, by allowing you to *parametrize* over a signature rather than a concrete implementation of a string type, and instantiate such an *indefinite library* whenever you want. This solves both problems:

1.  Because you are allowed to instantiate an indefinite library whenever you want, we can eagerly instantiate a `posix-indef` using `String` and ship it as `posix`, keeping backwards compatibility with all packages which are Backpack ignorant.
2.  At the same time, if packages depend directly on `posix-indef`, they themselves are parametrizable over a string type. Entire library ecosystems can defer the choice of string type to the end user, which on a sufficiently new version of GHC offers an backwards compatible way of adding support for new string types to a library. (I don't want to say, support *multiple* string types, because this is not necessarily a virtue in-and-of-itself.)

To this end, I would like to propose a string theory, for the base of GHC Haskell: namely the core boot libraries that are distributed with GHC today. These packages will set the tone for the eventual Backpackification of the rest of the ecosystem.

But first, what is it that we are parametrizing over? A string is not so simple...

### A digression on file paths (and OS strings)

File paths (`FilePath`) are an important form of `String` which aren't really Unicode strings at all. POSIX specifies that file paths can be arbitrary C strings, thus, code which decodes a file path as Unicode must be cognizant of the fact that the underlying `ByteString` could contain arbitrary, undecodable nonsense. To make matters worse, even the encoding can vary: on Windows file paths are encoded in UTF-16 (with unpaired surrogates, eek!), while in modern Linux environments the encoding is dictated by the locale (`base` uses `locale_charset` to determine how to interpret file paths; the locale is often UTF-8, but not always).

Thus, the definition `type FilePath = String` is very questionable indeed. There is an existing proposal, the [Abstract FilePath Proposal](https://ghc.haskell.org/trac/ghc/wiki/Proposal/AbstractFilePath) to turn `FilePath` into an abstract type, and not just a type synonym for `String`. Unfortunately, a change like this is a BC-breaking one, so it will take some time to implement, since GHC must first be taught to warn when `FilePath` is used as if it were a `String`, to help people find out that they are using it incorrectly.

Backpack offers a more decentralized way to move into the future: just define an *abstract signature* for `FilePath` to depend upon. The low level signature might look like this:

```
signature FilePath where

-- | File and directory names, whose precise
-- meaning is operating system dependent. Files can be opened, yielding a
-- handle which can then be used to operate on the contents of that file.
data FilePath

-- | A C string (pointer to an array of C characters terminated by NUL)
-- representing a file path, suitable for use with the operating system
-- C interface for file manipulation.  This exact type is architecture
-- dependent.
type CFilePath =
#ifdef mingw32_HOST_OS
        CWString
#else
        CString
#endif

withFilePath :: FilePath -> (CFilePath -> IO a) -> IO a
newFilePath  :: FilePath -> IO CFilePath
peekFilePath :: CFilePath -> IO FilePath
-- peekFilePath >=> newFilePath should be identity
-- (this is tricky to achieve if FilePath is a
-- Unicode-based type, like String)

```

And of course, you would want all of the `FilePath` [manipulation functions](https://hackage.haskell.org/package/filepath-1.4.1.0/docs/System-FilePath-Posix.html) that people use.

To maintain compatibility with the existing ecosystem, you would likely instantiate your library with `type FilePath = String`. But there is nothing stopping you from picking your own abstract `FilePath` type and using it instead.

File paths are not unique in this sense; there are other strings (such as the values of environment variables) which have similar properties: I've taken to calling these [OSStrings](https://doc.rust-lang.org/std/ffi/struct.OsString.html) (as they are called in Rust.)

### Axes of parametrization

With this in mind, there are three "string variants" any given library can be parametrized:

1.  They can be parametrized over `FilePath`, for modules which deal with the file system (e.g., [System.Posix.Directory](https://hackage.haskell.org/package/unix-2.7.2.0/docs/System-Posix-Directory.html))
2.  They can be parametrized over an `OSString`, because they deal with various operating system specific APIs (e.g., [System.Posix.Env](https://hackage.haskell.org/package/unix-2.7.2.0/docs/System-Posix-Env.html))
3.  They can be parametrized over a `String`, because, well, sometimes a string is just a string. (e.g., [Text.ParserCombinators.ReadP](https://hackage.haskell.org/package/base-4.9.0.0/docs/Text-ParserCombinators-ReadP.html))

Some libraries may be parametrized in multiple ways: for example, `readFile` needs to be parametrized over both `FilePath` and `String`.

### Split base (and friends) for Backpack

For technical reasons, Backpack cannot be used to parametrize specific *modules*; you have to parametrize over an entire library. So a side-effect of Backpack-ing the core libraries is that they will be split into a number of smaller libraries. Using module reexports, you can still keep the old libraries around as shims.

There are four GHC boot libraries which would most benefit from modularization on strings:

*   [base](https://hackage.haskell.org/package/base)
    *   base-io (System.IO and submodules; parametrized over FilePath and String)
    *   There are a few other modules which could be stringified, but the marginal benefit may not justify making a new package for each (Data.String, System.Console.GetOpt, Text.ParserCombinators.ReadP, Text.Printf). Each of these only needs to be parametrized over String.
    *   Control.Exception, Text.Read and Text.Show are explicit *non-goals*, they are too deeply wired into GHC at present to muck about with.
*   [unix](https://hackage.haskell.org/package/unix)
    *   unix-env (System.Posix.Env, parametrized over OSString)
    *   unix-fs (System.Posix.Directory, System.Posix.Files, System.Posix.Temp parametrized over FilePath)
    *   unix-process (System.Posix.Process, parametrized over FilePath and OSString)
*   [pretty](https://hackage.haskell.org/package/pretty) (parametrized over String; then GHC could use it rather than roll its own copy!)
*   [process](https://hackage.haskell.org/package/process) (parametrized over String, OSString and FilePath)

The naming scheme I propose is that, e.g., the package unix continues to be the package instantiated with old-fashioned Strings. Then unix-indef is a package which is uninstantiated (the user can instantiate it to what they want, or pass on the decision to their users). Some packages may choose to also provide shims of their package instantiated with specific types, e.g., `base-io-bytestring`, which would be `base-io` instantiated with `ByteString` rather than `String`, though these names could get to be quite long, so it's uncertain how useful this would be.