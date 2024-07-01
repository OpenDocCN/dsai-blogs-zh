<!--yml
category: 未分类
date: 2024-07-01 18:18:20
-->

# Spring 2010: A Random Walk : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/spring-2010-a-random-walk/](http://blog.ezyang.com/2010/05/spring-2010-a-random-walk/)

## Spring 2010: A Random Walk

Here at the eve of Spring 2010 term, I decided to run this little experiment on my laptop: what files had I modified within the last six months?

```
find . \( -path '*/.*' \) -prune -o -mtime -180 -print

```

The result was north of one-hundred-fifty thousand modified files. Here's the (slightly) abridged version:

*   LaTeX files for ["Adventures in Three Monads"](http://blog.ezyang.com/2010/01/adventures-in-three-monads/), an article that ran in the Monad Reader. Also, blackboard diagrams for my Advanced Typeclasses class I gave that IAP; I ended up not being able to get to the material I prepared for the Reader.
*   `valk.txt`, which contained notes for my Valkyrie Nethack character. I made my first ascension on March {24,25}th.
*   An Eclipse Java project that served as the jumpboard for my [HAMT experiments](http://blog.ezyang.com/2010/03/the-case-of-the-hash-array-mapped-trie/), under the purview of my undergraduate research project.
*   `htmlpurifier-web` and `htmlpurifier`, courtesy of the [HTML Purifier 4.1](http://htmlpurifier.org/) release I pushed within the last month. (I suspect there will be another release coming soon too.) This also meant new builds of PHP 5.2.11, 5.2.12, 5.2.13, 5.3.1 and 5.3.2 for my super-amazing PHP [multi-version farm](http://repo.or.cz/w/phpv.git). Note to self, next time, *exclude* the build directories from your automated backups, kthxbai.
*   A `qemu` checkout, in which I attempted to fix their broken DHCP code when the same MAC address requests two different IP addresses, gave up, and assigned static addresses to the virtual machines we were using to demo live process migration. Mmm... [6.828 final project](http://pdos.csail.mit.edu/6.828/).
*   A `hackage-server` and `holumbus` checkout, sprung from aborted dreams of making Holombus and Hackage cooperate to have up-to-the-minute index of all Haskell functions. I hear the Holumbus team has been making changes to make Hayoo be able to incrementally update its index.
*   Updates to tidy up `extort`, a membership dues tracking application written in Haskell, due to the recent change of leadership in the Assassins' Guild. During the replacement election, one of the suggested candidate questions was "Do you know Haskell." We'll see how long the program lasts...
*   An `abc` source directory, in which I flexed my C source-diving skills and searched for information on how to use the library. I may be working closely with it in my internship at Galois. Curiously enough, this roughly coincided with the SAT solver that was to be written for 6.005, as well as the study of SAT in my computation complexity class 6.045.
*   A `mit-scheme` checkout, in order to analyze their red-black tree implementation to figure out how easily it could be persisted (the answer was no, and I had to write my own implementation off of Okasaki's notes), and to figure out why `--batch-mode` didn't do [what it said on the tin](http://blog.ezyang.com/2010/04/art-code-math-and-mit-scheme/).
*   A `log4j` source tree, which I used for two of my Software Engineering 6.005 projects. It was mostly painless to use, and I highly recommend it if you're building software in Java.
*   Lots of test directories for `wizard` (note to self, backing those up is also a bad idea!) Some day, I'll [unleash this software](http://scripts.mit.edu/wizard/) on the world, but for now, it's usage is growing within the MIT sphere.

The really abridged version:

*   Languages of the half-year: Java, Haskell and Scheme
*   Biggest directory: I didn't count rigorously, but `linux`, `wizard` and `hackage` were pretty big.
*   Best filename: `./Dev/exploits/jessica_biel_naked_in_my_bed.c`

It's been a long random walk through lots of subjects, software and self-research. There is some trade-off from swapping subjects to really focus in after a month: on the one hand a month is really enough time to do anything *really* in the space (I feel much the same about my blog posts; they're an excuse to do the little experiments and segues, but nothing big), on the other hand it means I keep getting views of many specific subfields of computer science. With summer coming up soon, I will probably find another ambitious project to drive through my free time (or perhaps give some of my existing projects some of the love they need).