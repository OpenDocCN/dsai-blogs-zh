<!--yml
category: 未分类
date: 2024-07-01 18:18:09
-->

# Embracing Windows : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/09/embracing-window/](http://blog.ezyang.com/2010/09/embracing-window/)

## Embracing Windows

*Some things come round full circle.*

As a high schooler, I was a real Windows enthusiast. A budding programmer, I accumulated a complete development environment out of necessity, a mix of Cygwin, handwritten batch scripts, PuTTY, LogMeIn, a homegrown set of PHP build scripts and Notepad++. I was so devoted to the cause I even got a [single patch into Git](http://repo.or.cz/w/git.git/commit/36ad53ffee6ed5b7c277cde660f526fd8ce3d68f), for the purpose of making Git play nicely with plink on Windows. The setup worked, but it always felt like a patchwork of different components, all not quite seeing eye-to-eye with each other. When I discovered that Linux was able to offer me an unbelievably coherent development environment, I jumped ship and said goodbye to Windows.

*Some things come round full circle.* Windows has a way of coming back to you eventually. The [product I worked on over the summer](http://www.galois.com/technology/communications_security/cryptol) at Galois had to support Windows, and I consequently devoted days of effort getting my changes to build properly on Windows. I then went on to [hacking GHC](http://blog.ezyang.com/2010/08/interrupting-ghc/), and Simon Marlow asked me to implement the equivalent feature in Windows.

I’ve decided that I should stop shunning Microsoft Windows as the developer’s black sheep of the operating systems. Like it or not, Windows is here to stay; even if I never boot my laptop into Windows, as a developer it is good practice to think about and test my code on Windows. It might even be the case that Windows is a *perfectly reasonable* underlying platform to develop on.

There seem to be two reasons why developers might find targeting other platforms to be annoying:

*   They don’t have access to a computer running that operating system, which makes debugging the problems extremely annoying—after all, this is why a reproduceable test-case is the gold standard of bug reporting. We should have easy to access and easy to use build servers setup to let people play in these different environments. This involves putting down some money to buy the appropriate licenses, which open-source authors might be reluctant to do: people at places with site licenses might be able to help by donating boxes for these people to play in (the same way companies and universities donate disk space and bandwidth for mirrors).
*   They have to learn about another platform, with all of its intricacies and gotchas. On the one hand, this is annoying because “I already know how to do this in Unix, and now I have to spend N minutes to figure out how to do it on Windows, and spend another N minutes figuring out why it doesn’t work in some edge case.” On the other hand, learning a platform that does something you already know how to do can be kind of fun: you get to see different design decisions and develop multiple perspectives on the same problem, which I have found has always helped me out for problem solving.

There remain parts of Windows programming that I continue to have no interest in: for example, I find the vagaries of manifest files to be fairly uninteresting. But then again, I find packaging in Linux distributions to be uninteresting. Stop blaming Windows!