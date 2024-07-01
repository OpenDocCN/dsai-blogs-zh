<!--yml
category: 未分类
date: 2024-07-01 18:18:27
-->

# Hunting for constraints : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/02/hunting-for-constraints/](http://blog.ezyang.com/2010/02/hunting-for-constraints/)

```
> import Data.List
> import Control.Monad

```

The following question appeared as part of a [numbers-based puzzle](http://www.mit.edu/~puzzle/10/puzzles/2010/fun_with_numbers/) in the [2010 MIT Mystery Hunt](http://www.mit.edu/~puzzle/10/):

> His final level on [Wizard of Wor](http://hackage.haskell.org/package/dow) was equal to the smallest number that can be written as the sum of 4 non-zero squares in exactly 9 ways

I'd like to explore constraint search in Haskell to solve this problem. The hope is to find a (search) program that directly reflects the problem as posed, and also gives us an answer!

Because we are looking for the smallest number, it makes sense to start testing from a small number and start counting up. We'll assume that the answer to this question won't overflow Int.

Now, we need to test if it can be written as the sum of 4 non-zero squares in exactly 9 ways. This problem reduces to "how many ways can n be written as the sum of squares", which is another search problem.

We'll assume that 4+1+1+1 and 1+4+1+1 don't constitute distinct for the purposes of our nine squares. This results in the first piece of cleverness: if we impose a strict ordering on our squares, we once again get uniqueness.

We also need to bound our search space; while fair search can help to some degree with infinite failure, our implementation will be much simpler if we can do some early termination. A very simple condition to terminate on is if the sum of the squares exceeds the number we're looking for.

Considering the case where we are matching for x, and we have candidate roots a, b and c. Then, the maximum the remaining square can be is x - a^2 - b^2 - c^2, and the maximum value for d is the floor of the square root. Square roots are cheap, and we're using machine size integers, so things are good.

```
> floorSqrt :: Int -> Int
> floorSqrt = floor . sqrt . fromIntegral
>
> sumSquares :: [Int] -> Int
> sumSquares as = sum (map (^2) as)
>
> rootMax :: Int -> [Int] -> Int
> rootMax x as = floorSqrt (x - sumSquares as)

```

From there, we just write out the search for distinct sums of squares of a number:

```
> searchSumFourSquares :: Int -> [(Int, Int, Int, Int)]
> searchSumFourSquares x = do
>       a <- [1..(rootMax x [])]
>       b <- [a..(rootMax x [a])]
>       c <- [b..(rootMax x [a,b])]
>       d <- [c..(rootMax x [a,b,c])]
>       guard $ sumSquares [a,b,c,d] == x
>       return (a,b,c,d)

```

And from there, the solution falls out:

```
> search :: Maybe Int
> search = findIndex (==9) (map (length . searchSumFourSquares) [0..])

```

We cleverly use `[0..]` so that the index is the same as the number itself. Alternative methods might use tuples.