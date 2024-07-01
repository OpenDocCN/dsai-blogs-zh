<!--yml
category: 未分类
date: 2024-07-01 18:18:12
-->

# Buffered streams and iteratees : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/08/buffered-streams-and-iteratee/](http://blog.ezyang.com/2010/08/buffered-streams-and-iteratee/)

While attempting to figure out how I might explain lazy versus strict bytestrings in more depth without boring half of my readership to death, I stumbled upon a nice parallel between a standard implementation of buffered streams in imperative languages and iteratees in functional languages.

No self-respecting input/output mechanism would find itself without *buffering.* Buffering improves efficiency by grouping reads or writes together so that they can be performed as a single unit. A simple read buffer might be implemented like this in C (though, of course, with the static variables wrapped up into a data structure... and proper handling for error conditions in `read`...):

```
static char buffer[512];
static int pos = 0;
static int end = 0;
static int fd = 0;

int readChar() {
  if (pos >= end && feedBuf() == 0) {
    return EOF;
  }
  return (int) buffer[pos++];
}

int feedBuf() {
  pos = 0;
  end = read(fd, buffer, sizeof(buffer));
  assert(end != -1);
  return end;
}

```

The exported interface is `readChar`, which doles out a single `char` cast to an `int` every time a user calls it, but behind the scenes only actually reads from the input if it has run out of buffer to supply (`pos >= end`).

For most applications, this is good enough: the chunky underlying behavior is hidden away by a nice and simple function. Furthermore, our function is not *too* simple: if we were to read all of standard input into one giant buffer, we wouldn’t be able to do anything else until the `EOF` comes along. Here, we can react as the input comes in.

What would such a set of functions look like in a purely functional setting? One obvious difficulty is the fact that `buffer` is repeatedly mutated as we perform reads. In the spirit of persistence, we should very much prefer that our buffer not be mutated beyond when we initially fill it up. Making the buffer persistent means we also save ourselves from having to copy the data out if we want to hold onto it while reading in more data (you could call this zero copy). We can link buffers together using something simple: say, a linked list.

Linked lists eh? Let’s pull up the definition for lazy and strict ByteStrings (slightly edited for you, the reader):

```
data Strict.ByteString = PS !(ForeignPtr Word8) !Int !Int
data Lazy.ByteString = Empty | Chunk !Strict.ByteString Lazy.ByteString

```

In C, these would be:

```
struct strict_bytestring {
  char *pChar;
  int offset;
  int length;
}

struct lazy_bytestring {
  struct strict_bytestring *cur;
  int forced;
  union {
    struct lazy_bytestring *next;
    void (*thunk)(struct lazy_bytestring*);
  }
}

```

The Strict.ByteString is little more than a glorified, memory-managed buffer: the two integers track offset and length. Offset is an especially good choice in the presence of persistence: taking a substring of a string no longer requires a copy: just create a new strict ByteString with the offset and length set appropriately, and use the same base pointer.

So what is Lazy.ByteString? Well, it’s a glorified lazy linked list of strict ByteStrings—just read Chunk as Cons, and Empty as Null: the laziness derives from the lack of strictness on the second argument of `Chunk` (notice the lack of an exclamation mark, which is a strictness annotation). The laziness is why we have the `thunk` union and `forced` boolean in our `lazy_bytestring` struct: this API scribbles over the function pointer with the new `lazy_bytestring` when it is invoked. (This is not too different from how GHC does it; minus a layer of indirection or so.) If we ignore the laziness, this sounds a bit like the linked list of buffers we described earlier.

There is an important difference, however. A `Lazy.ByteString` is pure: we can’t call the original `read` function (a syscall, which makes it about as IO as you can get). So lazy ByteStrings are appropriate for when we have some pure computation (say, a Markov process) which can generate infinite amounts of text, but are lacking when it comes to buffering input.

“No problem!” you might say, “Just change the datatype to hold an `IO Lazy.ByteString` instead of a `Lazy.ByteString`:

```
data IO.ByteString = Empty | Chunk !Strict.ByteString (IO IO.ByteString)

```

But there’s something wrong about this datatype: nothing is stopping someone from invoking `IO IO.ByteString` multiple times. In fact, there’s no point in placing the IO operation in the `Chunk` value: due to the statefulness of file descriptors, the IO operation is the same code every time: `hReadByteString handle`. We’re back to handle-based IO.

The idea of `IO.ByteString` as a list is an important intuition, however. The key insight is this: who said that we have to give the list of IO actions to the user? Instead, *invert the control* so that the user doesn’t call the iteratee: the iteratee calls the user with the result of the IO. The user, in turn, can initiate other IO, or compose iteratees together (something we have not discussed) to stream from one iteratee to another.

At this point, I defer to Oleg’s [excellent annotated slides (PDF)](http://okmij.org/ftp/Haskell/Iteratee/IterateeIO-talk-notes.pdf) for further explanation of iteratees (no really, the slides are extremely well written), as well as the multitude of [iteratee](http://ianen.org/articles/understanding-iteratees/) [tutorials](http://cdsmith.wordpress.com/2010/05/23/iteratees-step-by-step-part-1/). My hope is that the emphasis on the “linked list of buffers” generated by IO operations directs some attention towards the fundamental nature of an iteratee: an abstraction on top of a list of IO actions.

To summarize:

*   Use *strict bytestrings* as a primitive for building more interesting structures that have buffers (though avoid reimplementing lazy bytestrings or iteratees). Use them when the amount of data is small, when all of it can be initialized at once, or when random access, slicing and other non-linear access patterns are important.
*   Use *lazy bytestrings* as a mechanism for representing infinite streams of data generated by pure computation. Consider using them when performing primarily operations well suited for lazy lists (`concat`, `append`, `reverse` etc). Avoid using them for lazy IO (despite what the module says on the tin).
*   Use *iteratees* for representing data from an IO source that can be incrementally processed: this usually means large datasets. Iteratees are especially well suited for multiple layers of incremental processing: they “fuse” automatically and safely.