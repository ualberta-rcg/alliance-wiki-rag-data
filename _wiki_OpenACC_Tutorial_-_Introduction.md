# OpenACC Tutorial - Introduction

## Learning Objectives

* Understand the difference between a CPU and an accelerator.
* Understand the difference between speed and throughput.
* Understand the steps to take to port an existing code to an accelerator.

## CPU vs Accelerator

Historically, computing has developed around Central Processing Units (CPU) that were optimized for sequential tasks. That is, they would complete only one compute operation during a given clock cycle. The frequency of these units steadily increased until about 2005, when the top speed of the high-end CPUs reached a plateau at around 4 GHz. Since then - for reasons well explained in [this article](link_to_article_needed) - the usual CPU clock frequency has barely moved, and is even now often lower than 4 GHz. Instead, manufacturers started adding multiple computation cores within a single chipset, opening wide the era of parallel computing.

Yet, even as of 2022, sequential tasks still run the fastest on CPUs:

1.  They have direct access to the main computer memory, which can be very large;
2.  Because of their very fast clock speed, they can run a small number of tasks very quickly.

But CPUs also have some weaknesses:

*   They have relatively low memory bandwidth;
*   They use cache mechanisms to mitigate the low bandwidth, but this means that cache misses are very costly;
*   And they also are rather power-hungry compared to accelerators.

Typical accelerators, such as GPUs or coprocessors, are highly parallel chipsets. They are made out of hundreds or thousands of relatively simple and low-frequency compute cores. Simply said, they are optimized for parallel computing. High-end GPUs usually have a few thousand compute cores. They also have a high bandwidth to access their own device memory. They present significantly more compute resources than high-end CPUs and provide much higher throughput and much better performance per watt. However, they embed a relatively low amount of memory and have low per-thread performance.

## Speed vs Throughput, Which is Best?

Speed is like a motorcycle: very fast, but only carries one person at a time. Throughput is like a train: slow, but carries hundreds of passengers in a single trip.

Depending on what kind of task you are trying to accomplish, you may want to use a high-speed device such as a CPU, or a high-throughput device such as an accelerator.

A **high-speed** device will accomplish a single task within a very short amount of time. This is probably what you want if you are trying to do a single sequential computation, such as the resolution of a one-dimensional differential equation. In real life, we could compare a high-speed device to a racing motorcycle or a racing car. It will bring a single passenger from point A to point B very quickly.

A **high-throughput** device will accomplish much more work, but in a longer amount of time. This is probably what you want if you are trying to solve a highly parallel problem. Examples of such tasks are numerous and include matrix operations, Fourier transforms, multidimensional differential equations, etc. In real life, we could compare a high-throughput device to a train or a bus. It will bring a lot of passengers from point A to point B, but in an admittedly longer time than a racing motorcycle or car.

## Porting Code to Accelerators

Porting a code to accelerators can be seen as a phase of an optimization process. A typical optimization process will have the following steps:

1.  Profile the code
2.  Identify bottlenecks
3.  Optimize the most significant bottleneck
4.  Validate the resulting code
5.  Start again from step 1

Much similarly, we can split the task of porting a code to accelerators into the following steps:

1.  Profile the code
2.  Identify parallelism within the bottlenecks
3.  Port the code
4.  Express parallelism to the compiler
5.  Express data movement
6.  Optimize loops
7.  Validate the resulting code
8.  Start again from step 1

OpenACC can be a rather descriptive language. This means that the programmer can tell the compiler that he thinks a given portion of the code can be parallelized and let the compiler figure out exactly how to do it. This is done by adding a few directives to the code (i.e., express parallelism in the above list). However, the quality of the compiler will greatly change the achieved performance. Even with the best compilers, there may be unnecessary data movement that needs to get taken out. This is what the programmer will do in the express data movement phase. Finally, the programmer may have information that is not available to the compiler which would allow him to achieve better performance by tuning the loops. This is what is done in the optimize loops step.

[Previous Unit](previous_unit_link_needed) | [Next Unit: Profiling](next_unit_link_needed)
