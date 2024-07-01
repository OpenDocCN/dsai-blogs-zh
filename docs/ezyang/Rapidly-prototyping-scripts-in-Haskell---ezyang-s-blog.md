<!--yml
category: 未分类
date: 2024-07-01 18:18:06
-->

# Rapidly prototyping scripts in Haskell : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/10/rapid-prototyping-in-haskell/](http://blog.ezyang.com/2010/10/rapid-prototyping-in-haskell/)

I’ve been having some vicious fun over the weekend hacking up a little tool called [MMR Hammer](http://github.com/ezyang/mmr-hammer) in Haskell. I won’t bore you with the vagaries of multimaster replication with Fedora Directory Server; instead, I want to talk about rapidly prototyping scripts in Haskell—programs that are characterized by a low amount of computation and a high amount of IO. Using this script as a case study, I’ll describe how I approached the problem, what was easy to do and what took a little more coaxing. In particular, my main arguments are:

1.  In highly specialized scripts, you can get away with not specifying top-level type signatures,
2.  The IO monad is the only monad you need, and finally
3.  You *can* and *should* write hackish code in Haskell, and the language will impose just the right amount of rigor to ensure you can clean it up later.

I hope to convince you that Haskell can be a great language for prototyping scripts.

*What are the characteristics of rapidly prototyping scripts?* There are two primary goals of rapid prototyping: to get it *working*, and to get it working *quickly.* There are a confluence of factors that feed into these two basic goals:

*   Your requirements are immediately obvious—the problem is an exercise of getting your thoughts into working code. (You might decide later that your requirements are wrong.)
*   You have an existing API that you want to use, which let’s you say “I want to set the X property to Y” instead of saying “I will transmit a binary message of this particular format with this data over TCP.” This should map onto your conception of what you want to do.
*   You are going to manually test by repeatedly executing the code path you care about. Code that you aren’t developing actively will in general not get run (and may fail to compile, if you have lots of helper functions). Furthermore, running your code should be fast and not involve a long compilation process.
*   You want to avoid shaving yaks: solving unrelated problems eats up time and prevents your software from working; better to hack around a problem now.
*   Specialization of your code for your specific use-case is good: it makes it easier to use, and gives a specific example of what a future generalization needs to support, if you decide to make your code more widely applicable in the future (which seems to happen to a lot of prototypes.)
*   You’re not doing very much computationally expensive work, but your logic is more complicated than is maintainable in a shell script.

*What does a language that enables rapid prototyping look like?*

*   It should be concise, and at the very least, not make you repeat yourself.
*   It should “come with batteries,” and at least have the important API you want to use.
*   It should be interpreted.
*   It should be well used; that is, what you are trying to do should exist somewhere in the union of what other people have already done with the language. This means you are less likely to run into bizarre error conditions in code that no one else runs.
*   It should have a fast write-test-debug cycle, at least for small programs.
*   The compiler should not get in your way.

*General prototyping in Haskell.* If we look at our list above, Haskell has several aspects that recommend it. GHC has a `runghc` command which allows you to interpret your script, which means for quick prototyping. Functional programming encourages high amounts of code reuse, and can be extremely concise when your comfortable with using higher-order functions. And, increasingly, it’s growing a rather large set of batteries. In the case of LDAP MMR, I needed a bindings for the OpenLDAP library, which [John Goerzen](http://hackage.haskell.org/package/LDAP) had already written. A great start.

*The compiler should not get in your way.* This is perhaps the most obvious problem for any newcomer to Haskell: they try to some pedestrian program and the compiler starts bleating at them with a complex type error, rather than the usual syntax error or runtime error. As they get more acquainted with Haskell, their mental model of Haskell’s type system improves and their ability to fix type errors improves.

The million dollar question, then, is how well do you have to know Haskell to be able to quickly resolve type errors? I argue, in the case of rapid prototyping in Haskell, *not much at all!*

One simplifying factor is the fact that the functions you write will usually *not* be polymorphic. Out of the 73 fully implemented functions in MMR Hammer, only six have inferred nontrivial polymorphic type signatures, all but one of these is only used single type context.

For these signatures, `a` is always `String`:

```
Inferred type: lookupKey :: forall a.
                            [Char] -> [([Char], [a])] -> [a]

Inferred type: lookupKey1 :: forall a.
                             [Char] -> [([Char], [a])] -> Maybe a

```

`m` is always `IO`, `t` is always `[String]` but is polymorphic because it’s not used in the function body:

```
Inferred type: mungeAgreement :: forall (m :: * -> *).
                                 (Monad m) =>
                                 LDAPEntry -> m LDAPEntry

Inferred type: replicaConfigPredicate :: forall t (m :: * -> *).
                                         (Monad m) =>
                                         ([Char], t) -> m Bool

```

`a` here is always `(String, String, String)`; however, this function is one of the few truly generic ones (it’s intended to be an implementation of `msum` for `IO`):

```
Inferred type: tryAll :: forall a. [IO a] -> IO a

```

And finally, our other truly generic function, a convenience debugging function:

```
Inferred type: debugIOVal :: forall b. [Char] -> IO b -> IO b

```

I claim that for highly specific, prototype code, GHC will usually infer fairly monomorphic types, and thus you don’t need to add very many explicit type signatures to get good errors. You may notice that MMR Hammer has almost *no* explicit type signatures—I argue that for monomorphic code, this is OK! Furthermore, this means that you only need to know how to *use* polymorphic functions, and not how to write them. (To say nothing of more advanced type trickery!)

*Monads, monads, monads.* I suspect a highly simplifying assumption for scripts is to avoid using any monad besides IO. For example, the following code *could* have been implemented using the Reader transformer on top of IO:

```
ldapAddEntry ldap (LDAPEntry dn attrs) = ...
ldapDeleteEntry ldap (LDAPEntry dn _ ) = ...
printAgreements ldap = ...
suspendAgreements ldap statefile = ...
restoreAgreements ldap statefile = ...
reinitAgreements ldap statefile = ...

```

But with only one argument being passed around, which was essentially required for any call to the API (so I would have done a bit of `ask` calling anyway), so using the reader transformer would have probably increased code size, as all of my LDAP code would have then needed to be lifted with `liftIO`.

Less monads also means less things to worry about: you don’t have to worry about mixing up monads and you can freely use `error` as a shorthand for bailing out on a critical error. In IO these get converted into exceptions which are propagated the usual way—because they are strings, you can’t write very robust error handling code, but hey, prototypes usually don’t have error handling. In particular, it’s good for a prototype to be brittle: to prefer to error out rather than to do some operation that may be correct but could also result in total nonsense.

Hanging lambda style also makes writing out code that uses bracketing functions very pleasant. Here are some example:

```
withFile statefile WriteMode $ \h ->
    hPutStr h (serializeEntries replicas)

forM_ conflicts $ \(LDAPEntry dn attrs) ->
    putStrLn dn

```

Look, no parentheses!

*Reaping the benefits.* Sometimes, you might try writing a program in another language for purely pedagogical purposes. But otherwise, if you know a language, and it works well for you, you won’t really want to change unless there are compelling benefits. Here are the compelling benefits of writing your code in Haskell:

*   When you’re interacting with the outside world, you will fairly quickly find yourself wanting some sort of concurrent execution: maybe you want to submit a query but timeout if it doesn’t come back in ten seconds, or you’d like to do several HTTP requests in parallel, or you’d like to monitor a condition until it is fulfilled and then do something else. Haskell makes doing this sort of thing ridiculously easy, and this is a rarity among languages that can also be interpreted.

*   Because you don’t have automatic tests, once you’ve written some code and manually verified that it works, you want it to stay working even when you’re working on some other part of the program. This is hard to guarantee if you’ve built helper functions that need to evolve: if you change a helper function API and forget to update all of its call sites, your code will compile but when you go back and try running an older codepath you’ll find you’ll have a bunch of trivial errors to fix. Static types make this go away. Seriously.

*   Haskell gives you really, really cheap abstraction. Things you might have written out in full back in Python because the more general version would have required higher order functions and looked ugly are extremely natural and easy in Haskell, and you truly don’t have to say very much to get a lot done. A friend of mine once complained that Haskell encouraged you to spend to much time working on abstractions; this is true, but I also believe once you’ve waded into the fields of Oleg once, you’ll have a better feel in the future for when it is and isn’t appropriate.

*   Rigorous NULL handling with Maybe gets you thinking about error conditions earlier. Many times, you will want to abort because you don’t want to bother dealing with that error condition, but other times you’ll want to handle things a little more gracefully, and the explicit types will always remind you when that is possible:

    ```
    case mhost of
        (Just host) -> do
            let status = maybe "no status found" id mstatus
            printf ("%-" ++ show width ++ "s : %s\n") host status
        _ -> warnIO ("Malformed replication agreement at " ++ dn)

    ```

*   Slicing and dicing input in a completely ad hoc way is doable and concise:

    ```
    let section = takeWhile (not . isPrefixOf "profile") . tail
                . dropWhile (/= "profile default") $ contents
        getField name = let prefix = name ++ " "
                        in evaluate . fromJust . stripPrefix prefix
                                    . fromJust . find (isPrefixOf prefix)
                                    $ section

    ```

    But at the same time, it’s not too difficult to rip out this code for a real parsing library for not too many more lines of code. This an instance of a more general pattern in Haskell, which is that moving from brittle hacks to robust code is quite easy to do (see also, static type system.)

*Some downsides.* Adding option parsing to my script was unreasonably annoying, and after staring at cmdargs and cmdlib for a little bit, I decided to roll my own with getopt, which ended up being a nontrivial chunk of code in my script anyway. I’m not quite sure what went wrong here, but part of the issue was my really specialized taste in command line APIs (based off of Git, no less), and it wasn’t obvious how to use the higher level libraries to the effect I wanted. This is perhaps witnessed by the fact that most of the major Haskell command line applications also roll their own command parser. More on this on another post.

Using LDAP was also an interesting exercise: it was a fairly high quality library that worked, but it wasn’t comprehensive (I ended up submitting a patch to support `ldap_initialize`) and it wasn’t battle tested (it had no workaround for a longstanding bug between OpenLDAP and Fedora DS—more on that in another post too.) This is something that gets better with time, but until then expect to work closely with upstream for specialized libraries.