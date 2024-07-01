<!--yml
category: 未分类
date: 2024-07-01 18:17:24
-->

# Plan 9 mounts and dependency injection : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/11/plan-9-mounts-and-dependency-injection/](http://blog.ezyang.com/2012/11/plan-9-mounts-and-dependency-injection/)

“Everything is a file.” [1] This was the design philosophy taken to its logical extreme in [Plan 9](http://en.wikipedia.org/wiki/Plan_9_from_Bell_Labs). Any interface you could imagine was represented as a file. Network port, pixel buffers, kernel interfaces—all were unified under a common API: the file operations (`open`, `read`, `write`...) Plan 9 used this to eliminate most of its system calls: it had only thirty-nine, in contrast to modern Linux's sprawling three hundred and twenty-six.

When I first heard of Plan 9, my first thought was, “But that’s cheating, right?” After all, they had reduced the number of syscalls but increased the number of custom files: complexity had merely been shifted around. But one of [my labmates](https://plus.google.com/107333307248367605396/about) gave me a reason why this was still useful: per-process mountpoints. These mountpoints meant that I could give each process their own view of the filesystem—usually the same, but sometimes with some vital differences. Suppose that I wanted to tunnel the network connection of one of my applications: this application would be accessing the network through some file, so I instead could mount a network filesystem to the network files of another system, and transparently achieve proxying without any cooperation from my application. [2]

Let’s step back for a moment and put on our programming language hats. Suppose that a file is an abstract data type, and the syscall interface for manipulating files is the interface for this data type. What are mounts, in this universe? Another [friend of mine](https://plus.google.com/116034118081904229710/about) pointed out the perfectly obvious analogy:

> Files : Mounts :: Abstract Data Types : Dependency Injection

In particular, the mount is a mechanism for modifying some local namespace, so that when a file is requested, it may be provided by some file system completely different to what the process might have expected. Similarly, dependency injection specifies a namespace, such that when an object is requested, the concrete implementation may be completely different to what the caller may have expected.

The overall conclusion is that when developers implemented dependency injection, they were reimplementing Plan 9’s local mounts. Is your dependency injection hierarchical? Can you replace a hierarchy (`MREPL`), or mount your files before (`MBEFORE`) or after (`MAFTER`) an existing file system? Support runtime changes in the mount? Support lexical references (e.g. dot-dot `..`) between entities in the hierarchy? I suspect that existing dependency injection frameworks could learn a bit from the design of Plan 9\. And in Haskell, where it seems that people are able to get much further without having to create a dependency injection framework, do these lessons map back to the design of a mountable file system? *I wonder.*

* * *

[1] Functional programmers might be reminded of a similar mantra, “Everything is a function.”

[2] For the longest time, Linux did not provide per-process mount namespaces, and even today this feature is not available to unprivileged users—Plan 9, in contrast, had this feature available from the very beginning to all users. There is also the minor issue where per-process mounts are actually a big pain to work with in Linux, primarily, I dare say, due to the lack of appropriate tools to assist system administrators attempting to understand their applications.