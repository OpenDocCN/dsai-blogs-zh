<!--yml
category: 未分类
date: 2024-07-01 18:17:42
-->

# Facebook support for BarnOwl : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/07/facebook-support-for-barnowl/](http://blog.ezyang.com/2011/07/facebook-support-for-barnowl/)

## Facebook support for BarnOwl

This one's for the MIT crowd. This morning, I finished my [Facebook module for BarnOwl](https://github.com/ezyang/barnowl) to my satisfaction (my satisfaction being asynchronous support for Facebook API calls, i.e. no more random freezing!) Getting it to run on Linerva was a bit involved, however, so here is the recipe.

1.  Setup a local CPAN installation using the [instructions at sipb.mit.edu](http://sipb.mit.edu/doc/cpan/), using `local::lib`. Don’t forget to add the setup code to `.bashrc.mine`, not `.bashrc`, and then source them. Don't forget to follow prerequisites: otherwise, CPAN will give a lot of prompts.
2.  Install all of the CPAN dependencies you need. For the Facebook module, this means `Facebook::Graph` and `AnyEvent::HTTP`. I suggest using `notest`, since `Any::Moose` seems to fail a harmless test on Linerva. `Facebook::Graph` fails several tests, but don't worry about it since we'll be using a pre-packaged version. If you want to use other modules, you will need to install them in CPAN as well.
3.  Clone BarnOwl to a local directory (`git clone git://github.com/ezyang/barnowl.git barnowl`), `./autogen.sh`, `configure` and `make`.
4.  Run using `./barnowl`, and then type the command `:facebook-auth` and follow the instructions!

Happy Facebooking!

*Postscript.* I am really, really surprised that there is not a popular imperative language that has green threads and pre-emptive scheduling, allowing you to actually write code that looks blocking, although it uses an event loop under the hood. Maybe it’s because being safe while being pre-emptive is hard...

*Known bugs.* Read/write authentication bug has been fixed. We seem to be tickling some bugs in BarnOwl's event loop implementation, which is causing crashing on the order of day (making it tough to debug). Keep a backup instance of BarnOwl handy.