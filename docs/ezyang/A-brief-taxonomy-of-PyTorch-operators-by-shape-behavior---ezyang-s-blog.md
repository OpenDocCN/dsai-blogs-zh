<!--yml
category: 未分类
date: 2024-07-01 18:16:52
-->

# A brief taxonomy of PyTorch operators by shape behavior : ezyang’s blog

> 来源：[http://blog.ezyang.com/2020/05/a-brief-taxonomy-of-pytorch-operators-by-shape-behavior/](http://blog.ezyang.com/2020/05/a-brief-taxonomy-of-pytorch-operators-by-shape-behavior/)

I've recently been working on a revamp of how we specify tensor shape formulas in PyTorch. As part of this process, I classified *every single operator* in PyTorch by its shaping behavior; yes, that's all 1364 of them (this includes each variant of an operator; e.g., inplace and `out=` keyword variants). During the process, I tried to come up with categories to help classify what operators did. One of the surprises from the process was discovering that shaping behaviors that I previously thought were uncommon, actually showed up a bit more often than one might have expected.

These categories are interesting in their own right and can be used to help understand how PyTorch's API fits together. Here are all the categories I devised.

**TensorIterator** (505, e.g., add, sum) operators are PyTorch's bread and butter; these operators do pointwise operations and reductions and support [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html) and [type promotion](https://pytorch.org/docs/stable/tensor_attributes.html). The name [TensorIterator](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorIterator.h) refers to an internal abstraction we have in PyTorch for implementing these operations; you can read more about it on [the wiki](https://github.com/pytorch/pytorch/wiki/How-to-use-TensorIterator) and in [this blog post](https://labs.quansight.org/blog/2020/04/pytorch-tensoriterator-internals/). TensorIterator is a real workhorse in PyTorch: the plurarity (though not majority) of operators are implemented in this way! Note that this category includes some functions that used equivalent, legacy functionality (but did not exactly use TensorIterator).

**Fixed** (273, e.g., convolution, addbmm) operators are operators which only work on a fixed number of dimensions. This assumption makes writing efficient kernels a lot easier, as indexing math is simple with fixed dimensionality. (For example, [TensorAccessor](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/TensorAccessor.h) is an internal class which lets you view a tensor at fixed dimensionality known at compile time). Sometimes, the first dimension is treated as a batch dimension, but not always (unfortunately, I didn't distinguish these cases in my dataset). Some fixed operators actually support multiple dimensions, but only a fixed number of them; for example, because we only support 1-3D convolutions, this counts as fixed. (Compare with this FeatureBatched, below!)

**N-Dimensional** (107, e.g., squeeze, index_add, tensordot) operators are operators which work generically on tensors of arbitrary dimensionality. These are the operations for which it is difficult to write generic shaping rules for in symbolic form, as you need a language that can talk about list manipulations. An important subclass of N-dimensional operators are **Identity** (42, e.g., clone, contiguous; not included in the count above) operators work over arbitrary dimensionality, but they always return a tensor with the same size as their input. Another subclass are **Flatten** (11, e.g. take, bucketize) operators which accept tensors of any dimensionality, but always treat them as 1D tensors internally.

**Composite** (95, e.g., kl_div, isfinite) operators are implemented in other operators, and don't themselves have shape checking (instead, they rely on the operations they call to check shapes). Note this category is probably a bit underreported, as in some cases when it was obvious what the underlying behavior of an operator was, I classified the operator as that category, rather than Composite.

**Batched** (94, e.g., nll_loss, adaptive_avg_pool2d) operators are like fixed dimensionality operators, except they accept an arbitrary number of batch dimensions at their beginning. Many fixed operators should be batched operators; others cannot be converted into batched operators without introducing ambiguity as to where the batch dimensions end. Compare these with **FeatureBatched** (19, e.g., batch_norm, embedding) operators, which are like batched operators, but rather than accept batch dimensions at the beginning, they accept an arbitrary number of feature dimensions at the end.

**Factory** (90, e.g., empty) operators produce new tensors without having any tensor inputs.

**Trivial** (59, e.g., size, is_floating_point) operators aren't actual tensor operations, but ways to return non-Tensor information or access internal data structures

**Sparse** (40) operators are special because their size calculations take account of both dense and sparse dimensions.

**Dynamic** (15, e.g., unique) operators produce outputs whose shapes depend on the data of their input tensors

**Variadic** (14, e.g., cat) operators take multiple input tensors; similar to n-dimensional operations they are difficult to capture symbolic

You can take a look at the full data set at [https://docs.google.com/spreadsheets/d/e/2PACX-1vQQFW0T_bucT5KZn0BHYTC1KYhkL6ZMG5ZxQWc6UmAkHUDYpqkpzXnsb59uv2TB0Jgc1Q6qO63bx6WQ/pubhtml](https://docs.google.com/spreadsheets/d/e/2PACX-1vQQFW0T_bucT5KZn0BHYTC1KYhkL6ZMG5ZxQWc6UmAkHUDYpqkpzXnsb59uv2TB0Jgc1Q6qO63bx6WQ/pubhtml)