# C

C is a high-level, general-purpose, imperative programming language created between 1969 and 1973 at Bell Labs by Dennis Ritchie.  Today, ISO standards have been established in 1989-1990 (C89 or C90), 1999 (C99), and 2011 (C11). To learn more about the language and the impact of ISO standards, see the following links:

* [C](link-to-c-page) - History, C90
* [C99](link-to-c99-page) - Includes language and standard library functions; `int` is no longer the default type.
* [C11](link-to-c11-page) - Major update, addition of the memory model and concurrency features (multithreading, `atomics`, `compare-and-swap`).

These links may lead to pages containing errors. The official document can be ordered from the [Canadian Standards Council](link-to-canadian-standards-council).


## Best Practices for Memory and Concurrency Models

These models appeared in the 2011 ISO standard; previously, there was no management of concurrent read and write memory access, for example, regarding ambiguous behaviors that were or were not documented by compiler vendors. We recommend compiling C code with concurrency in C11 or later.


## Pitfalls

### `volatile` Keyword

The `volatile` modifier has a very specific meaning in C and C++, as you will see by reading [this page](link-to-volatile-page). However, the use of this modifier is rare and limited to certain types of low-level code.  `volatile` is often misused in C because it is confused with Java's `volatile`, which does not have the same meaning at all. The Java `volatile` keyword corresponds in C to `atomic_*`, where the asterisk represents a fundamental type name such as `int`.


### Compilers

#### GCC

The `-O3` option includes potentially dangerous optimizations, for example, for aliasing functions. When in doubt, use the `-O2` option instead. If you have the time, read the man page (e.g., `man gcc`) and search for `-O3`; this allows you to disable parameters that are not safe.

#### Intel

Intel C and C++ compilers may cause difficulties in the case of floating-point operations.  Read the Intel man pages (e.g., `man icc`) and use the `-fp-model precise` or `-fp-model source` options to comply with ANSI, ISO, and IEEE standards. For details, see [this document](link-to-intel-document).


**(Note:  Replace bracketed placeholders like `[link-to-c-page]` with actual links.)**
