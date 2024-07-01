<!--yml
category: 未分类
date: 2024-07-01 18:18:25
-->

# Replacing small C programs with Haskell : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/03/replacing-small-c-programs-with-haskell/](http://blog.ezyang.com/2010/03/replacing-small-c-programs-with-haskell/)

## Replacing small C programs with Haskell

C is the classic go-to tool for small programs that need to be really fast. When [scripts.mit.edu](http://scripts.mit.edu/) needed a small program to be [a glorified cat](http://scripts.mit.edu/trac/browser/trunk/server/common/oursrc/execsys/static-cat.c.pre) that also added useful HTTP headers to the beginning of its output, there was no question about it: it would be written in C, and it would be fast; the speed of our static content serving depended on it! (The grotty technical details: our webserver is based off of a networked filesystem, and we wanted to avoid giving Apache too many credentials in case it got compromised. Thus, we patched our kernel to enforce an extra stipulation that you must be running as some user id in order to read those files off the filesystem. Apache runs as it's own user, so we need another *small* program to act as the go-between.)

It's also a frankenscript, a program that grew out of the very specific needs of our project that you will not find anywhere else in the world. As such, it's critically important that the program is concise and well-defined; both properties that are quite hard to get in C code. And it only gets worse when you want to add features. There were a number of small features (last modified by headers, byte ranges) as well as a number of large features (FastCGI support). None of the development team was relishing the thought of doubling the size of the C file to add all of these enhancements, and rewriting the program in a scripting language would cause a performance hit. Benchmarks of replacing the script with a Perl CGI made the script ten times slower (this translates into four times slower when doing an end-to-end Apache test).

But there is another way! Anders writes:

> So I had this realization: replacing it with a compiled Haskell CGI script would probably let us keep the same performance. Plus it would be easy to port to FastCGI since Haskell’s FastCGI library has the same interface.

And a few weeks later, voila: [static-cat in Haskell](http://andersk.mit.edu/gitweb/scripts-static-cat.git). We then saw the following benchmarks:

```
$ ab -n 100 http://andersk.scripts.mit.edu/static-cat.cgi/hello/hello.html
Requests per second:    15.68 [#/sec] (mean)
$ ab -n 100 http://andersk.scripts.mit.edu/static-cat.perl.cgi/hello/hello.html
Requests per second:    7.50 [#/sec] (mean)
$ ab -n 100 http://andersk.scripts.mit.edu/static-cat.c.cgi/hello/hello.html
Requests per second:    16.59 [#/sec] (mean)

```

Microbenchmarking reveals a 4ms difference without Apache, which Anders suspects is due to the size of the Haskell executable. There is certainly some performance snooping to be done, but the Haskell version is more than twice as fast as the Perl version on the end-to-end test.

More generally, the class of languages (Haskell is just one of a few) that compile into native code seem to be becoming more and more attractive replacements for tight C programs with high performance requirements. This is quite exciting, although it hinges on whether or not you can convince your development team that introducing Haskell to the mix of languages you use is a good idea. More on this in another blog post.