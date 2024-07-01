<!--yml
category: 未分类
date: 2024-07-01 18:17:46
-->

# Tail recursion makes your loops cleaner : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/05/tail-recursion-makes-your-loops-cleaner/](http://blog.ezyang.com/2011/05/tail-recursion-makes-your-loops-cleaner/)

## Tail recursion makes your loops cleaner

Recursion is one of those things that functional programming languages shine at—but it seems a bit disappointing that in many cases, you have to convert your beautiful recursive function back into iterative form. After all, iteration is what imperative languages do best, right?

Actually, explicitly tail-recursive functions in functional programming languages can be fairly beautiful: in fact, in the cases of complicated loops, they can be even prettier than their imperative counterparts. Take this midpoint line-drawing algorithm as an example:

```
circleMidpoint d r = go 0 (-r) k0
    where k0 = 5 - 4 * r
          x1 = ceiling (fromIntegral r / sqrt 2)
          go x y k | x > x1    = return ()
                   | k > 0     = d (x,y) >> go (x+1) (y+1) (k+8*x+8*y+20)
                   | otherwise = d (x,y) >> go (x+1)  y    (k+8*x+12)

```

There are three loop variables: `x`, `y` and `k`, and depending on various conditions, some of them get updated in different ways. `x` is a bog-standard loop variable; ye old C-style `for` loop could handle it just fine. But `y` and `k` are updated differently depending on some loop conditions. But since they’re parameters to the `go` helper function, it’s always clear what the frequently changing variables are. You lose that nice structure in the imperative translation:

```
// global variables and loop variables are all mixed together
int k = 5 - 4 * r;
int y = -r;
int x1 = ceil(r/sqrt(2));
for (int x = 0; x <= x1; x++) { // only x is obviously an index var
  draw(x, y);
  if (k > 0) {
    y++;
    k += 8*x + 8*y + 20;
  } else {
    k += 8*x + 12;
  }
  // does it ever make sense for any code to live here?
}

```

I’ve also managed to introduce a bug in the process...