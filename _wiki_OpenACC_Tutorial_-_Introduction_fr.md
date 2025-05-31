# OpenACC Tutorial - Introduction

## Learning Objectives

*   Understand the difference between a CPU and an accelerator.
*   Understand the difference between speed and throughput.
*   Understand the steps to port existing code to an accelerator.

## Difference Between CPU and Accelerator

Historically, computing development revolved around the CPU (central processing unit).  However, CPUs could only complete one calculation per clock cycle. CPU clock frequencies steadily increased until around 2005, plateauing at approximately 4 GHz.  Since then, there has been minimal increase; many clocks today remain below 4 GHz. An article by Pär Persson Mattsson explains why. Manufacturers opted to add computing cores to a single circuit, paving the way for parallel computing.

In 2022, sequential tasks are still faster with CPUs:

*   They have direct access to often large memory.
*   They can execute a small number of tasks very quickly due to clock speed.

However, CPUs have drawbacks:

*   Low memory bandwidth.
*   Cache mechanisms compensate for low bandwidth, but cache misses are costly.
*   They are more energy-intensive than accelerators.

A typical accelerator, such as a GPU or coprocessor, is a highly parallel chipset composed of hundreds or thousands of low-frequency, relatively simple computing cores. They are optimized for parallel computing. High-end GPUs typically have hundreds of computing cores and high memory bandwidth (device memory). They offer significantly more computing resources than the most powerful CPUs, providing better throughput and performance per watt.  However, they have relatively little memory and low per-thread performance.


## Prioritize Speed or Throughput?

Speed is like a motorcycle; it's fast but carries only one passenger. Throughput is like a train; it's slower but carries hundreds of passengers in a single trip.

Depending on the task, a CPU offers the advantage of speed, while an accelerator is preferred for its throughput.

A high-speed component can accomplish a task very quickly, desirable for a single sequential calculation, such as a first-order differential equation.  Such a high-speed component is comparable to a racing motorcycle or car: the passenger is moved very quickly from point A to point B.

A high-throughput component accomplishes much more work but takes longer, desirable for solving highly parallel problems. Examples include matrix operations, Fourier transforms, and multidimensional differential equations. This type of component is comparable to a train or bus: multiple passengers are moved from point A to point B, but certainly more slowly than with a motorcycle or car.


## Porting Code to an Accelerator

This can be considered a phase in the optimization process. A typical case involves these steps:

1.  Profile the code.
2.  Identify bottlenecks.
3.  Replace the most significant bottleneck with optimized code.
4.  Validate the output code.
5.  Return to step 1.

Porting code to an accelerator involves these steps:

1.  Profile the code.
2.  Locate parallelism in bottlenecks.
3.  Port the code.
4.  Describe the parallelism to the compiler.
5.  Describe the data flow to the compiler.
6.  Optimize loops.
7.  Validate the result.
8.  Return to step 1.

OpenACC can be a descriptive language. The programmer can indicate to the compiler which code sections to parallelize, letting the compiler do the work. This is done by adding directives to the code (see point 3.1 above, "Describe the parallelism to the compiler"). However, compiler quality significantly impacts performance. Even with the best compilers, some data movements may need to be eliminated, which the programmer can do at point 3.2, "Describe the data flow to the compiler". Finally, if the programmer knows how to improve performance by adjusting loops, they will inform the compiler at point 3.3, "Optimize loops".


[^Return to the beginning of the tutorial]: #openacc-tutorial-introduction

[Next page, Profilers]: #profilers
