<!--yml
category: 未分类
date: 2024-07-01 18:17:45
-->

# Pinpointing space leaks in big programs : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/06/pinpointing-space-leaks-in-big-programs/](http://blog.ezyang.com/2011/06/pinpointing-space-leaks-in-big-programs/)

What is the biggest possible Haskell program that you could try debugging a space leak in? One very good choice is GHC, weighing in nearly a 100k lines of code (though, thankfully, 25% of that figure is comments.) Today, I’m going to describe one such space leak that I have fixed in GHC. This is not a full story: my code was ultimately the culprit, so not covered is how to debug code you did not write. However, I still like to think this story will cover some of the major points:

I really like this case because I had to do all of these things in order to pinpoint and ultimately fix this space leak. I hope you do too!

### Setting up a test-case

When I finally got around to attacking this bug seriously, the first thing I wanted to do was make a reduced test-case of `Parser.hs`, effectively the input file that was causing the out-of-memory. Why not test on `Parser.hs` directly?

*   Big inputs result in lots of data, and if you don’t really know what you’re looking for lots of data is overwhelming and confusing. The answer might be really obvious, but if there is too much cruft you’ll miss it.
*   This was a file that was OOM-ing a machine with 2GB of RAM. Space is time, and Haskell programs that use this much memory take correspondingly longer to run. If I was looking to make incremental changes and re-test (which was the hope), waiting half an hour between iterations is not appealing.
*   It was a strategy that had worked for me in the past, and so it seemed like a good place to start collecting information.

Actually, I cheated: I was able to find another substantially smaller test file lying around in GHC’s test suite that matched the heap profile of GHC when run on `Parser.hs`, so I turned my attentions there.

In your case, you may not have a “smaller” test case lying around. In that case, you will need to reduce your test-case. Fortunately, inputs that are not source programs tend to be a lot easier to reduce!

*   Binary search for a smaller size. Your null hypothesis should be that the space leak has been caused strictly more data, so if you delete half of your input data, the leak should still be present, just not as severe. Don’t bother with anything sophisticated if you can chop and re-test.
*   Sometimes a space leak is caused by a specific type of input, in which case deleting one half of the input set may cause the leak to go away. In this case, the first thing you should test is the other half of the input set: the culprit data is probably there, and you can continue binary searching on that chunk of code. In the worst case scenario (removing either half causes the leak to go away), buckle down and start selectively removing lines that you believe are "low risk." If you remove a single line of data and the leak goes away, you have some very good data about what may actually be happening algorithmically.
*   In the case of input data that has dependencies (for example, source code module imports), attempt to eliminate the dependencies first by re-creating them with stub data.

In the best case situation, this process will only take a few minutes. In the worst case situation, this process may take an hour or so but will yield good insights about the nature of the problem. Indeed, after I had gotten my new test-case, I reduced it even further, until it was a nice compact size that I could include in a blog post:

```
main = print $ length [
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),
  ([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0),([0,0,0],0)]

```

A plain heap profile of looks very similar to the bigger Parser case, and even if it’s a different space leak, it’s still worth fixing.