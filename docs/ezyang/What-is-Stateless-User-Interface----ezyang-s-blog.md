<!--yml
category: 未分类
date: 2024-07-01 18:17:06
-->

# What is Stateless User Interface? : ezyang’s blog

> 来源：[http://blog.ezyang.com/2015/11/what-is-stateless-user-interface/](http://blog.ezyang.com/2015/11/what-is-stateless-user-interface/)

The essence of stateless user interface is that actions you take with a program should not depend on implicit state. Stateless interfaces are easier to understand, because an invocation of a command with some arguments will *always* do the same thing, whereas in a stateful interface, the command may do some different than it did yesterday, because that implicit state has changed and is influencing the meaning of your program.

This philosophy is something any Haskeller should intuitively grasp... but Cabal and cabal-install today fail this ideal. Here are some examples of statefulness in Cabal today:

1.  Running `cabal install`, the built packages are installed into a "package database", which makes them available for use by GHC.
2.  Running `cabal install`, the choice of what packages and versions to install depends on the state of the local package database (the current solver attempts to reuse already installed software) and the state of the remote package repository (which says what packages and versions are available.)
3.  Running `./Setup configure` saves a `LocalBuildInfo` to `dist/setup-config`, which influences further `Setup` commands (`build`, `register`, etc.)

Each of these instances of state imposes complexity on users: how many times do you think you have (1) blown away your local package database because it was irreversibly wedged, (2) had your project stop building because the dependency solver started picking too new version of packages, or (3) had Cabal ask you to reconfigure because some feature wasn't enabled?

State has cost, but it is not here for no reason:

1.  The package database exists because we don't want to have to rebuild all of our packages from scratch every time we want to build something (indeed, this is the whole point of a package manager);
2.  The solver depends on the local package database because users are impatient and want to avoid building new versions packages before they can build their software;
3.  The solver depends on the remote package repository because developers and users are impatient and want to get new releases to users as quickly possible;
4.  The configure caches its information because a user doesn't want to wait to reconfigure the package every time they try to build a package they're working on.

In the face of what is seemingly an inherently stateful problem domain, can stateless user interface prevail? Of course it can.

Sometimes the state is merely being used as a *cache*. If a cache is blown away, everything should still work, just more slowly. The package database (reason 1) and configuration cache (reason 4) both fall under this banner, but the critical mistake today's Cabal makes is that if you delete this information, things do *not* "just work". There must be sufficient information to rebuild the cache; e.g., the configuration cache should be supplemented with the actual input to the configure step. (Sometimes, separation of concerns means you simply cannot achieve this. What is `ghc` to do if you ask it to use the not-in-cache lens package?) Furthermore, the behavior of a system should not vary depending on whether or not the cached data is present or not; e.g., the solver (reason 2) should not make different (semantically meaningful) decisions based on what is cached or not.

Otherwise, it must be possible to explicitly manage the state in question: if the state is a remote package repository (reason 3), there must be a way to pin against some state. (There's a tool that does this and it's called Stack.) While sometimes necessary, explicit state complicates interface and makes it harder to describe what the system can do. Preferably, this state should be kept as small and as centralized as possible.

I don't think anything I've said here is particularly subtle. But it is something that you need to specifically think about; otherwise, you will be seduced by the snare of stateful interface. But if you refuse the siren call and put on the hair shirt, your users will thank you much more for it.

*Acknowledgments.* These thoughts are not my own: I have to give thanks to Johan Tibell, Duncan Coutts, and Simon Marlow, for discussions which communicated this understanding to me. Any mistakes in this article are my own. This is not a call to action: the Cabal developers recognize and are trying to fix this, see this [hackathon wiki page](https://github.com/haskell/cabal/wiki/Hackathon2015) for some mutterings on the subject. But I've not seen this philosophy written out explicitly anywhere on the Internet, and so I write it here for you.