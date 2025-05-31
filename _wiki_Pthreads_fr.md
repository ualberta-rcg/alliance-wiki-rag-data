# Pthreads

This page is a translated version of the page Pthreads and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-url)


## Introduction

The term `pthreads` comes from POSIX threads, one of the first parallelization techniques. Like other tools using threads, pthreads is used in a shared memory context and therefore usually on a single node where the number of active threads is limited to the CPU cores available on that node. Pthreads are used in several programming languages, but mostly in C. In Fortran, thread parallelization is preferably done with OpenMP. In C++, the tools from the Boost library resulting from the C11 standard are better suited to object-oriented programming.

The pthreads library served as the basis for subsequent parallelization approaches, including OpenMP. Pthreads can be seen as a set of primitive tools offering basic parallelization functionalities, unlike user-friendly and high-level APIs like OpenMP. In the pthreads model, threads are generated dynamically to execute so-called lightweight subroutines that perform operations asynchronously; these threads are then destroyed after rejoining the main process. Since all threads in the same program reside in the same memory space, it is easy to share data using global variables, unlike a distributed approach like MPI; however, any modification to shared data risks creating race conditions.

To parallelize a program with pthreads or any other technique, it is important to consider the program's ability to run in parallel, which we will call its scalability. After parallelizing your software and being satisfied with its quality, we recommend performing a scalability analysis to understand its performance.


## Compilation

To use the functions and data structures associated with pthreads in your C program, you must include the header file `pthread.h` and compile the program with a flag to link with the pthreads library.

```bash
[name@server ~]$ gcc -pthread -o test threads.c
```

The number of threads for the program is defined by one of the following methods:

* Used as an argument in a command line.
* Entered via an environment variable.
* Encoded in the source file (this does not allow adjusting the number of threads at runtime).


## Creating and Destroying Pthreads

To parallelize an existing sequential program with pthreads, we use a programming model where threads are created by a parent, perform part of the work, and then rejoin the parent. The parent is either the sequential master thread or one of the other slave threads.

The `pthread_create` function creates new threads with these four arguments:

1. The unique identifier for the new thread.
2. The set of thread attributes.
3. The C function that the thread executes when it is started (the launch routine).
4. The argument of the launch routine.


```c
// File: thread.c
#include <stdio.h>
#include <pthread.h>

const long NT = 12;

void *task(void *thread_id) {
    long tnumber = (long)thread_id;
    printf("Hello World from thread %ld\n", 1 + tnumber);
}

int main(int argc, char **argv) {
    int success;
    long i;
    pthread_t threads[NT];

    for (i = 0; i < NT; ++i) {
        success = pthread_create(&threads[i], NULL, task, (void *)i);
        if (success != 0) {
            printf("ERROR: Unable to create worker thread %ld successfully\n", i);
            return 1;
        }
    }

    for (i = 0; i < NT; ++i) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
```

In this example, the thread index (from 0 to 11) is passed as an argument; the `task` function is therefore executed by each of the 12 threads. Note that the `pthread_create` function does not block the master thread, which continues to execute the `main` function after the creation of each thread. Once the 12 threads are created, the master thread enters the second `for` loop and calls the blocking function `pthread_join`: the master thread then waits for the 12 slave threads to finish executing the `task` function and then rejoin the master thread. This simple example illustrates the basic operation of a POSIX thread: the master thread creates a thread by assigning it a function to execute and then waits for the created thread to finish this function, then rejoins the master thread.

By running this code several times in a row, you will probably notice a variation in the order in which the slave threads say hello, which is predictable since they run asynchronously. Each time the program is executed, the 12 threads respond simultaneously to the `printf` function and it is never the same thread that wins the race.


## Synchronizing Data Access

In a real program, slave threads must read and in some cases modify data to accomplish their tasks. This data is usually a set of global variables of various types and dimensions; concurrent read and write access by multiple threads must therefore be synchronized to avoid race conditions, i.e., cases where the program's result depends on the order in which slave threads access the data. If a parallel program must give the same result as its serial version, race conditions must not occur.

The simplest and most used way to control concurrent access is the lock; in the context of pthreads, the locking mechanism is the mutex (for mutual exclusion). Variables of this type are assigned to only one thread at a time. After reading or modifying, the thread disables the lock. The code between the variable call and the time it is disabled is executed exclusively by this thread. To create a mutex, you must declare a global variable of type `pthread_mutex_t`. This variable is initialized by the `pthread_mutex_init` function. At the end of the program, the resources are unlocked by the `pthread_mutex_destroy` function.


```c
// File: thread_mutex.c
#include <stdio.h>
#include <pthread.h>

const long NT = 12;
pthread_mutex_t mutex;

void *task(void *thread_id) {
    long tnumber = (long)thread_id;
    pthread_mutex_lock(&mutex);
    printf("Hello World from thread %ld\n", 1 + tnumber);
    pthread_mutex_unlock(&mutex);
}

int main(int argc, char **argv) {
    int success;
    long i;
    pthread_t threads[NT];

    pthread_mutex_init(&mutex, NULL);

    for (i = 0; i < NT; ++i) {
        success = pthread_create(&threads[i], NULL, task, (void *)i);
        if (success != 0) {
            printf("ERROR: Unable to create worker thread %ld successfully\n", i);
            pthread_mutex_destroy(&mutex);
            return 1;
        }
    }

    for (i = 0; i < NT; ++i) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&mutex);
    return 0;
}
```

In this example based on the content of the `thread.c` file above, access to the standard output channel is serialized as it should be with a mutex. The call to `pthread_mutex_lock` performs the blocking, that is, the thread will wait indefinitely for the mutex to become available. It is necessary to ensure that the code does not cause another blocking since the mutex must eventually become available. This is particularly problematic in a real program that contains several mutex variables controlling access to different global data structures.

In the case of the non-blocking alternative `pthread_mutex_trylock`, the non-zero value is immediately produced if the mutex is not fulfilled, thus indicating that the mutex is busy. It is also necessary to ensure that there is no superfluous code inside the serialized block; since this code is executed serially, it must be as concise as possible so as not to impair parallelism in the execution of the program.

More subtle synchronization is possible with the read/write lock `pthread_rwlock_t`. This tool allows simultaneous reading of a variable by several threads, but behaves like a standard mutex, i.e., no other thread has access to this variable (in reading or writing). As with the mutex, the `pthread_rwlock_t` lock must be initialized before use and destroyed when it is no longer needed. A thread obtains a read lock with `pthread_rwlock_rdlock` and a write lock with `pthread_rwlock_wrlock`. In both cases, the lock is destroyed with `pthread_rwlock_unlock`.

Another tool allows several threads to act on the same condition, for example, to wait for slave threads to be solicited for a task. This is a condition variable, expressed as follows: `pthread_cond_t`. Like the mutex or the read/write lock, the condition variable must be initialized before use and destroyed when it is no longer needed. To use this condition variable, a mutex must control access to the variables that affect the condition. A thread waiting for a condition locks the mutex and calls the `pthread_cond_wait` function with the condition variable and the mutex as arguments. The mutex is destroyed atomically with the creation of the condition variable whose result is awaited by the thread; other threads can then lock the mutex, either to wait for the condition or to modify one or more variables, which will modify the condition.


```c
// File: thread_condition.c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h> // for atoi

const long NT = 2;
pthread_mutex_t mutex;
pthread_cond_t ticker;
int workload;

void *task(void *thread_id) {
    long tnumber = (long)thread_id;
    if (tnumber == 0) {
        pthread_mutex_lock(&mutex);
        while (workload <= 25) {
            pthread_cond_wait(&ticker, &mutex);
        }
        printf("Thread %ld: incrementing workload by 15\n", 1 + tnumber);
        workload += 15;
        pthread_mutex_unlock(&mutex);
    } else {
        int done = 0;
        do {
            pthread_mutex_lock(&mutex);
            workload += 3;
            printf("Thread %ld: current workload is %d\n", 1 + tnumber, workload);
            if (workload > 25) {
                done = 1;
                pthread_cond_signal(&ticker);
            }
            pthread_mutex_unlock(&mutex);
        } while (!done);
    }
}

int main(int argc, char **argv) {
    int success;
    long i;
    pthread_t threads[NT];

    if (argc < 2) {
        printf("Usage: %s <initial_workload>\n", argv[0]);
        return 1;
    }

    workload = atoi(argv[1]);
    if (workload > 25) {
        printf("Initial workload must be <= 25, exiting...\n");
        return 0;
    }

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&ticker, NULL);

    for (i = 0; i < NT; ++i) {
        success = pthread_create(&threads[i], NULL, task, (void *)i);
        if (success != 0) {
            printf("ERROR: Unable to create worker thread %ld successfully\n", i);
            pthread_mutex_destroy(&mutex);
            return 1;
        }
    }

    for (i = 0; i < NT; ++i) {
        pthread_join(threads[i], NULL);
    }

    printf("Final workload is %d\n", workload);
    pthread_cond_destroy(&ticker);
    pthread_mutex_destroy(&mutex);
    return 0;
}
```

In this example, two slave threads modify the value of the integer `workload` whose initial value must be less than or equal to 25. The first thread locks the mutex and waits because `workload <= 25`; the condition variable `ticker` is created and the mutex is destroyed. The second thread can then execute the loop, which increments the value of `workload` by three at each iteration. At each increment, the second thread checks if the value of `workload` is greater than 25; if so, the thread calls `pthread_cond_signal` to signal the waiting thread that the condition is met. Once the signal is received by the first thread, the second thread sets the loop exit condition, primes the mutex and disappears with `pthread_join`. Meanwhile, the first thread being awakened, it increments the value of `workload` by 15 and leaves the `task` function. When all slave threads are rejoined, the master thread prints the final value of `workload` and the program ends.

In general, in a real program where several threads are waiting for a condition variable, the `pthread_cond_broadcast` function signals all waiting threads that the condition is met. In this context, `pthread_cond_signal` would alert only one thread at random and the other threads would remain waiting.


## Learn More

For more information on pthreads, on optional arguments for the various functions (the parameters used on this page use the default argument NULL) and on advanced topics, we recommend David Butenhof's book, *Programming with POSIX Threads* or the excellent [Lawrence Livermore National Laboratory tutorial](link-to-llnl-tutorial).


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Pthreads/fr&oldid=135740](https://docs.alliancecan.ca/mediawiki/index.php?title=Pthreads/fr&oldid=135740)"
