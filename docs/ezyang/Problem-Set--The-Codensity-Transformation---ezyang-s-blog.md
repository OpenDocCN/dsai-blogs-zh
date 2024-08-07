<!--yml

category: 未分类

date: 2024-07-01 18:17:38

-->

# 问题集：[Codensity 变换](http://blog.ezyang.com/2012/01/problem-set-the-codensity-transformation/)：ezyang's 博客

> 来源：[`blog.ezyang.com/2012/01/problem-set-the-codensity-transformation/`](http://blog.ezyang.com/2012/01/problem-set-the-codensity-transformation/)

你是否曾经想过*codensity 变换*，这是一种惊人通用的技巧，可以加速某些类型的单子执行，但始终无法理解论文或 Edward Kmett 在这个主题上的博文？

不用再找了：以下是学习这个变换如何工作的*问题集*。

这些练习的理念是让你熟悉与 codensity 变换相关的类型，通过使用这些类型来指导你自己找到唯一可能的实现方式。我们从经典的叶树具体实例开始热身，然后泛化到所有自由单子上（如果你不知道是什么也别担心：我们会定义它并给出一些热身练习）。

有经验在延续传递风格中编写可能会很有用，尽管在实践中这归结为“听从类型！”

更多解答和评论可在 Janis Voigtlander 的论文 "[计算在自由单子上的渐近改进](http://www.iai.uni-bonn.de/~jv/mpc08.pdf)" 中找到。

要了解更多，请参阅 Edward Kmett 的优秀文章，进一步概括了这个概念：

如果有需求，我可以为练习添加提示部分。

```
{-# LANGUAGE Rank2Types, MultiParamTypeClasses, FlexibleInstances #-}
import Prelude hiding (abs)

_EXERCISE_ = undefined

-----------------------------------------------------------------------------
-- Warmup: Hughes lists
-----------------------------------------------------------------------------

-- Experienced Haskellers should feel free to skip this section.

-- We first consider the problem of left-associative list append.  In
-- order to see the difficulty, we will hand-evaluate a lazy language.
-- For the sake of being as mechanical as possible, here are the
-- operational semantics, where e1, e2 are expressions and x is a
-- variable, and e1[e2/x] is replace all instances of x in e1 with e2.
--
--        e1 ==> e1'
--   ---------------------
--     e1 e2 ==> e1' e2
--
--   (\x -> e1[x]) e2 ==> e1[e2/x]
--
-- For reference, the definition of append is as follows:
--
--      a ++ b = foldr (:) b a
--
-- Assume that, on forcing a saturated foldr, its third argument is
-- forced, as follows:
--
--                e1 ==> e1'
--    -----------------------------------
--      foldr f e2 e1 ==> foldr f e2 e1'
--
--  foldr f e2 (x:xs) ==> f x (foldr f e2 xs)
--
-- Hand evaluate this implementation by forcing the head constructor,
-- assuming 'as' is not null:

listsample as bs cs = (as ++ bs) ++ cs

-- Solution:
--
--        (as ++ bs) ++ cs
--      = foldr (:) cs (as ++ bs)
--      = foldr (:) cs (foldr (:) bs as)
--      = foldr (:) cs (foldr (:) bs (a:as'))
--      = foldr (:) cs (a : foldr (:) b as')
--      = a : foldr (:) cs (foldr (:) bs as')
--
-- Convince yourself that this takes linear time per append, and that
-- processing each element of the resulting tail of the list will also
-- take linear time.

-- We now present Hughes lists:

type Hughes a = [a] -> [a]

listrep :: Hughes a -> [a]
listrep = _EXERCISE_

append :: Hughes a -> Hughes a -> Hughes a
append = _EXERCISE_

-- Now, hand evaluate your implementation on this sample, assuming all
-- arguments are saturated.

listsample' a b c = listrep (append (append a b) c)

-- Solution:
--
--        listrep (append (append a b) c)
--      = (\l -> l []) (append (append a b) c)
--      = (append (append a b) c) []
--      = (\z -> (append a b) (c z)) []
--      = (append a b) (c [])
--      = (\z -> a (b z)) (c [])
--      = a (b (c []))
--
-- Convince yourself that the result requires only constant time per
-- element, assuming a, b and c are of the form (\z -> a1:a2:...:z).
-- Notice the left-associativity has been converted into
-- right-associative function application.

-- The codensity transformation operates on similar principles.  This
-- ends the warmup.

-----------------------------------------------------------------------------
-- Case for leafy trees
-----------------------------------------------------------------------------

-- Some simple definitions of trees

data Tree a = Leaf a | Node (Tree a) (Tree a)

-- Here is the obvious monad definition for trees, where each leaf
-- is substituted with a new tree.

instance Monad Tree where
    return = Leaf
    Leaf a   >>= f = f a
    Node l r >>= f = Node (l >>= f) (r >>= f)

-- You should convince yourself of the performance problem with this
-- code by considering what happens if you force it to normal form.

sample = (Leaf 0 >>= f) >>= f
    where f n = Node (Leaf (n + 1)) (Leaf (n + 1))

-- Let's fix this problem.  Now abstract over the /leaves/ of the tree

newtype CTree a = CTree { unCTree :: forall r. (a -> Tree r) -> Tree r }

-- Please write functions which witness the isomorphism between the
-- abstract and concrete versions of trees.

treerep :: Tree a -> CTree a
treerep = _EXERCISE_

treeabs :: CTree a -> Tree a
treeabs = _EXERCISE_

-- How do you construct a node in the case of the abstract version?
-- It is trivial for concrete trees.

class Monad m => TreeLike m where
    node :: m a -> m a -> m a
    leaf :: a -> m a
    leaf = return

instance TreeLike Tree where
    node = Node

instance TreeLike CTree where
    node = _EXERCISE_

-- As they are isomorphic, the monad instance carries over too.  Don't
-- use rep/abs in your implementation.

instance Monad CTree where
    return = _EXERCISE_
    (>>=)  = _EXERCISE_ -- try explicitly writing out the types of the arguments

-- We now gain efficiency by operating on the /abstracted/ version as
-- opposed to the ordinary one.

treeimprove :: (forall m. TreeLike m => m a) -> Tree a
treeimprove m = treeabs m

-- You should convince yourself of the efficiency of this code.
-- Remember that expressions inside lambda abstraction don't evaluate
-- until the lambda is applied.

sample' = treeabs ((leaf 0 >>= f) >>= f)
    where f n = node (leaf (n + 1)) (leaf (n + 1))

-----------------------------------------------------------------------------
-- General case
-----------------------------------------------------------------------------

-- Basic properties about free monads

data Free f a = Return a | Wrap (f (Free f a))
instance Functor f => Monad (Free f) where
    return = _EXERCISE_
    (>>=)  = _EXERCISE_ -- tricky!

-- Leafy trees are a special case, with F as the functor. Please write
-- functions which witness this isomorphism.

data F a = N a a

freeFToTree :: Free F a -> Tree a
freeFToTree = _EXERCISE_

treeToFreeF :: Tree a -> Free F a
treeToFreeF = _EXERCISE_

-- We now define an abstract version of arbitrary monads, analogous to
-- abstracted trees.  Witness an isomorphism.

newtype C m a = C { unC :: forall r. (a -> m r) -> m r }

rep :: Monad m => m a -> C m a
rep = _EXERCISE_

abs :: Monad m => C m a -> m a
abs = _EXERCISE_

-- Implement the monad instance from scratch, without rep/abs.

instance Monad (C m) where
    return = _EXERCISE_
    (>>=)  = _EXERCISE_ -- also tricky; if you get stuck, look at the
                        -- implementation for CTrees

-- By analogy of TreeLike for free monads, this typeclass allows
-- the construction of non-Return values.

class (Functor f, Monad m) => FreeLike f m where
    wrap :: f (m a) -> m a

instance Functor f => FreeLike f (Free f) where
    wrap = Wrap

instance FreeLike f m => FreeLike f (C m) where
    -- Toughest one of the bunch. Remember that you have 'wrap' available for the
    -- inner type as well as functor and monad instances.
    wrap = _EXERCISE_

-- And for our fruits, we now have a fully abstract improver!

improve :: Functor f => (forall m. FreeLike f m => m a) -> Free f a
improve m = abs m

-- Bonus: Why is the universal quantification over 'r' needed?   What if
-- we wrote C r m a = ...?  Try copypasting your definitions for that
-- case.

```
