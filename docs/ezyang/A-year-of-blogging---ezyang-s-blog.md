<!--yml
category: 未分类
date: 2024-07-01 18:18:00
-->

# A year of blogging : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/12/a-year-of-blogging/](http://blog.ezyang.com/2010/12/a-year-of-blogging/)

## A year of blogging

Here is to celebrate a year of blogging. Thank you all for reading. It was only [a year ago](http://blog.ezyang.com/2009/12/iron-blogger/) that I first opened up shop under the wings of Iron Blogger. Iron Blogger has mostly disintegrated at this point, but I’m proud to say that this blog has not, publishing thrice a week, every week (excepting that one time I missed a post and made it up with a bonus post later that month), a bet that I made with myself and am happy to have won.

Where has this blog gone over the year? According to Google Analytics, here were the top ten most viewed posts:

1.  [Graphs not grids: How caches are corrupting young algorithms designers and how to fix it](http://blog.ezyang.com/2010/07/graphs-not-grids/)
2.  [You could have invented zippers](http://blog.ezyang.com/2010/04/you-could-have-invented-zippers/)
3.  [Medieval medicine and computers](http://blog.ezyang.com/2010/11/medieval-medicine-and-computers/)
4.  [Databases are categories](http://blog.ezyang.com/2010/06/databases-are-categories/)
5.  [Design Patterns in Haskell](http://blog.ezyang.com/2010/05/design-patterns-in-haskel/)
6.  [Static Analysis for everday (not-PhD) man](http://blog.ezyang.com/2010/06/static-analysis-mozilla/)
7.  [MVC and purity](http://blog.ezyang.com/2010/07/mvc-and-purity/)
8.  [Day in the life of a Galois intern](http://blog.ezyang.com/2010/08/day-in-the-life-of-a-galois-intern/)
9.  [Replacing small C programs with Haskell](http://blog.ezyang.com/2010/03/replacing-small-c-programs-with-haskell/)
10.  [How to use Vim’s textwidth like a pro](http://blog.ezyang.com/2010/03/vim-textwidth/)

There are probably a few obscure ones that are my personal favorites, but I’ve written so many at this point it’s a little hard to count: including this post, I will have published 159 posts, totaling somewhere around 120,000 words. (This figure includes markup, but for comparison, a book is about 80,000 words. Holy cow, I’ve written a book and a half worth of content. I don’t really feel like a better writer though—this may be because I’ve skimped on the “revising” bit of the process.)

This blog will go on a brief hiatus for the month of January. Not because I wouldn’t be able to produce posts over the holidays (given the chance, I probably would... in fact, this was a kind of hard decision to make) but because I should spend a month concentrating the bulk of my free time on stuff other than blogging. Have a happy New Years, and see you in February!

*Postscript.* Here is the SQL query I used to count:

```
select
  sum( length(replace(post_content, '  ', '')) - length(replace(post_content, ' ', ''))+1)
from wp_posts
where post_status = 'publish';

```

There's probably a more accurate way of doing it, but I was too lazy to write out the script.