# Java

Java is a high-level, object-oriented programming language created in 1995 by Sun Microsystems (acquired by Oracle in 2009).  The central goal of Java is that software written in this language obeys the principle "write once, run anywhere" and is very easily portable across multiple operating systems because Java source code compiles into octal code (bytecode) that can be executed on a Java environment (JVM for Java virtual machine); different architectures and platforms can therefore constitute a uniform environment. This characteristic makes Java a popular language in certain contexts, particularly for learning programming. Even if the emphasis is not on performance, there are ways to increase execution speed and the language has gained some popularity among scientists in fields such as life sciences, from which, for example, the genomic analysis tools GATK from the Broad Institute originate. The purpose of this page is not to teach the Java language, but to provide advice and suggestions for its use on the Alliance clusters.


The Alliance offers several Java environments via the `module` command. In principle, you will have only one Java module loaded at a time. The main commands associated with Java modules are:

*   `java`: to launch a Java environment;
*   `javac`: to call the Java compiler that converts a Java source file into bytecode.

Java software is frequently distributed as JAR files with the suffix `.jar`. To use Java software, use the command:

```bash
[name@server ~]$ java -jar file.jar
```

## Parallelism

### Threads

Java allows programming with threads, thus eliminating the need for interfaces and libraries like OpenMP, pthreads, and Boost that are necessary with other languages. The main Java object for handling concurrency is the `Thread` class; it can be used by providing a `Runnable` method to the standard `Thread` class or by defining the `Thread` class as a subclass, as demonstrated here:

**File: `thread.java`**

```java
public class HelloWorld extends Thread {
    public void run() {
        System.out.println("Hello World!");
    }
    public static void main(String args[]) {
        (new HelloWorld()).start();
    }
}
```

This approach is generally the simplest, but it has the disadvantage of not allowing multiple inheritance; the class that implements concurrent execution cannot therefore subclass another potentially more useful class.

### MPI

The `MPJ Express` library is often used to obtain MPI-type parallelism.


## Pitfalls

### Memory

A Java instance expects to have access to all the physical memory of a node, while the scheduler or an interpreter might impose its own (often different) limits depending on the submission script specifications or the limits of the connection node. In a shared resource environment, these limits ensure that finite-capacity resources such as memory and CPU cores are not exhausted by one task to the detriment of another.

When a Java instance is launched, it sets the value of two parameters according to the amount of physical memory rather than the amount of available memory as follows:

*   Initial heap size: 1/64 of physical memory
*   Maximum heap size: 1/4 of physical memory

In the presence of a large amount of physical memory, this 1/4 value can easily exceed the limits imposed by the scheduler or an interpreter, and Java may stop and produce messages such as:

```
Could not reserve enough space for object heap
There is insufficient memory for the Java Runtime Environment to continue.
```

However, these two parameters can be explicitly controlled by either of the following statements:

```bash
java -Xms256m -Xmx4g -version
```

or

```bash
java -XX:InitialHeapSize=256m -XX:MaxHeapSize=4g -version
```

To see all the command-line options that the instance will execute, use the `-XX:+PrintCommandLineFlags` flag as follows:

```bash
$ java -Xms256m -Xmx4g -XX:+PrintCommandLineFlags -version
-XX:InitialHeapSize=268435456 -XX:MaxHeapSize=4294967296 -XX:ParallelGCThreads=4 -XX:+PrintCommandLineFlags -XX:+UseCompressedOops -XX:+UseParallelGC
```

You can use the `JAVA_TOOL_OPTIONS` environment variable to configure execution options rather than specifying them on the command line. This is useful when multiple calls are launched or when a program is called by another Java program. Here is an example:

```bash
[name@server ~]$ export JAVA_TOOL_OPTIONS="-Xms256m -Xmx2g"
```

During execution, the program issues a diagnostic message similar to `Picked up JAVA_TOOL_OPTIONS`; this indicates that the options have been taken into account.

Remember that the Java instance itself creates a memory usage reserve. We recommend that the per-task limit be set to 1 or 2 GB more than the value of the `-Xmx` option.


### Garbage Collection (GC)

Java uses the automatic Garbage Collection process to identify variables with invalid values and return the associated memory to the operating system. By default, the Java instance uses a parallel GC and determines a number of GC threads equal to the number of CPU cores of the node, whether or not the Java task is multithreaded. Each of the GC threads consumes memory. In addition, the amount of memory consumed by the GC threads is proportional to the amount of physical memory. We therefore strongly recommend that you have a number of GC threads equal to the number of CPU cores you request from the scheduler in the submission script, for example, `-XX:ParallelGCThreads=12`. You can also use the sequential GC with the `-XX:+UseSerialGC` option, whether or not the task is parallel.


### `volatile` Keyword

The meaning of this keyword is very different from that of the same term used in C/C++ programming. The value of a Java variable with this attribute is always read directly from main memory and always written directly to main memory; any modification to the variable will therefore be visible to all other threads. In some contexts, however, `volatile` is not enough to prevent race conditions, and `synchronized` is necessary to maintain program consistency.


## References

OAKS, Scott and Henry Wong, *Java Threads: Understanding and Mastering Concurrent Programming*, 3rd edition, O'Reilly, 2012.
