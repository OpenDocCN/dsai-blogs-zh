<!--yml
category: 未分类
date: 2024-07-01 18:18:25
-->

# Mutation sleuthing in Python : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/03/mutation-sleuthing-in-python/](http://blog.ezyang.com/2010/03/mutation-sleuthing-in-python/)

Python is a language that gives you a lot of rope, in particular any particular encapsulation scheme is only weakly enforced and can be worked around by a sufficiently savvy hacker. I fall into the "my compiler should stop me from doing stupid things" camp, but I'll certainly say, dynamic capabilities sure are convenient. But here's the rub: *the language must show you where you have done something stupid.*

In this case, we'd like to see when you have improperly gone and mutated some internal state. You might scoff and say, "well, I know when *I* change *my* state", but this is certainly not the case when you're debugging an interaction between two third party libraries that you did not write. Specifically I should be able to point at a variable (it might be a local variable, a global variable, or a class/instance attribute) and say to Python, "tell me when this variable changes." When the variable changes, Python should tell me who changed the variable (via a backtrace) and what the variable changed to. I should be able to say, "tell me when this variable changed to this value."

Well, here is a small module that does just that: [mutsleuth](http://github.com/ezyang/mutsleuth/blob/master/mutsleuth.py). Import this module and install the watcher by passing `mutsleuth.watch` an expression that evaluates to the variable you'd like to check.

Here's an example: suppose I have the following files:

`good.py`:

```
b = "default value"

```

`evil.py`:

```
import good
good.b = "monkey patch monkey patch ha ha ha"

```

`test.py`:

```
import mutsleuth
mutsleuth.watch("good.b")

import good
import evil

```

When you run test.py, you'll get the following trace:

```
ezyang@javelin:~/Dev/mutsleuth$ python test.py
Initialized by:
  File "test.py", line 5, in <module>
    import evil
  File "/home/ezyang/Dev/mutsleuth/good.py", line 1, in <module>
    b = "good default value"
Replaced by:
  File "test.py", line 5, in <module>
    import evil
  File "/home/ezyang/Dev/mutsleuth/evil.py", line 2, in <module>
    good.b = "monkey patch monkey patch ha ha ha"

```

There are a few caveats:

*   Tracing doesn't start until you enter another local scope, whether by calling a function or importing a module. For most larger applications, you will invariably get this scope, but for one-off scripts this may not be the case.
*   In order to keep performance tolerable, we only do a shallow comparison between instances, so you'll need to specifically zoom in on a value to get real mutation information about it.

Bug reports, suggestions and improvements appreciated! I went and tested this by digging up an old bug that I would have loved to have had this module for (it involved logging code being initialized twice by two different sites) and verified it worked, but I haven't tested it "cold" yet.

Hat tip to [Bengt Richter](http://mail.python.org/pipermail/python-list/2002-September/164261.html) for suggesting this tracing originally.