<!--yml
category: 未分类
date: 2024-07-01 18:18:28
-->

# Cute macro tricks in the kernel : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/02/kernel-macro-tricks/](http://blog.ezyang.com/2010/02/kernel-macro-tricks/)

## Cute macro tricks in the kernel

A classic stylistic tip given to C programmers is that inline functions should be preferred over macros, when possible. This advice stems from the fact that a macro and an inline function can achieve the same effect, but the inline function also gets type checking.

As it turns out, you *can* achieve static type checking with macros, if you're willing to resort to the same cute trick that this following snippet from the Linux kernel uses:

```
#define module_param_named(name, value, type, perm) \
        param_check_##type(name, &(value)); \
        module_param_call(name, param_set_##type, param_get_##type, &value, perm); \
        __MODULE_PARM_TYPE(name, #type)

```

Hmm... I wonder what that `param_check_##type` call is all about. Digging through a few more macro definitions, we see:

```
#define __param_check(name, p, type) \
        static inline type *__check_##name(void) { return(p); }

```

So there you go. A throw-away inline function named `__check_##name` enforces that `p` is the same type as `type`. A comment is also given, explaining what's going on:

```
/* The macros to do compile-time type checking stolen from Jakub
   Jelinek, who IIRC came up with this idea for the 2.4 module init code. */

```