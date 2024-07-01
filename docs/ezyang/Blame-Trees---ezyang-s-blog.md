<!--yml
category: 未分类
date: 2024-07-01 18:17:18
-->

# Blame Trees : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/08/blame-trees/](http://blog.ezyang.com/2013/08/blame-trees/)

## Blame Trees

I just presented *Blame Trees* at the [13th Algorithms and Data Structures Symposium](http://www.wads.org/). Blame trees are a functional data structure which support an efficient merge operation by incorporating information about the “blame” (think `git blame`) of any given part of the structure. It’s a theory paper, so the constant factors are not so good, but the asymptotics are much better than traditional merge algorithms used by modern VCSes.

This was joint work with [David A. Wilson](http://web.mit.edu/dwilson/www/), [Pavel Panchekha](http://pavpanchekha.com/) and [Erik D. Demaine](http://erikdemaine.org/). You can view the [paper](http://ezyang.com/papers/demaine13-blametrees.pdf) or check out the [slides.](http://ezyang.com/slides/ezyang13-blametrees-slides.pdf) I also have a slightly older version of the talk recorded on [YouTube (20 minutes)](http://youtu.be/f8e-QE6Gus8) which I had used to help get feedback from my out-of-town collaborators before actually giving the talk. Thanks also to David Mazières for giving useful comments on the presentation in person.