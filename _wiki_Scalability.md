# Scalability

Other languages: English, français

In the context of parallel programming, **scalability** refers to a program's capacity to efficiently utilize added computing resources, i.e., CPU cores.  Naively, doubling the CPU cores should halve calculation time; however, this is rarely the case. Performance gains depend on the problem's nature, the algorithm/program, the hardware (memory and network), and the number of CPU cores used.  Therefore, when using a parallel program on a cluster, a scalability analysis is recommended. This involves testing the software with a fixed problem while varying the number of CPU cores (e.g., 2, 4, 8, 16, 32, 64 cores). The runtime for each core count is recorded and plotted.

## Why is Scalability Often Worse Than Expected?

Two main reasons explain this:

1.  **Serial Operations:** Not all operations parallelize, leaving a percentage of the program's execution serial. This limits parallel efficiency.  If a serial program takes an hour, with six minutes (10%) spent on unparallelizable operations, even infinite cores won't reduce runtime below six minutes. Ideally, this "serial percentage" decreases as the problem size increases.

2.  **Parallel Overhead:** Parallelization requires inter-process communication and synchronization. This "parallel overhead" increases with the number of processes, typically as a power of the number of cores:

   $T_c \propto n^\alpha$

   where $\alpha > 1$.

   If the scientific part's runtime is equally divided among cores (excluding the serial part):

   $T_s = A + B/n$

   The total runtime:

   $T = T_s + T_c = A + B/n + Cn^\alpha$

   (with A, B, and C being positive real numbers depending on the cluster, program, and test problem) will be dominated by the parallel overhead as $n \to \infty$.  If A and B are much larger than C, the runtime vs. CPU core count plot resembles the accompanying figure (figure not included in original text).

   The crucial point is that runtime initially decreases, reaches a minimum (e.g., at n ≈ 22), and then *increases* with added processes.  Scalability analysis determines the optimal CPU core count (4, 128, 1024, etc.) for a given problem and cluster.

## Choosing a Test Problem

The test problem should be relatively small for quick testing but large enough to represent a production run. A 30-60 minute runtime on one or two cores is a good choice; anything under ten minutes is likely too short.  For weak scaling analysis (see below), the problem size should be easily and ideally continuously increased.

"Embarrassingly parallel" problems have negligible parallel overhead (C ≈ 0).  An example is analyzing 500 independent files, each producing a single number.  Synchronization and communication are unnecessary, allowing perfect scaling up to the number of files.


## Strong Scaling vs. Weak Scaling

"Scaling," unqualified, usually means strong scaling. However, weak scaling might be more relevant depending on the goal.

*   **Strong Scaling:**  The problem size is fixed while the CPU core count increases. Ideally, runtime decreases proportionally to the core count increase.

*   **Weak Scaling:** The problem size increases proportionally to the CPU core count. Ideally, runtime remains constant.


### Strong Scaling

A fixed problem is used while increasing the CPU core count. Linear scaling is ideal (doubling cores halves runtime).  An example:

| Cores | Run Time (s) | Efficiency (%) |
|---|---|---|
| 2 | 2765 | N/A |
| 4 | 1244 | 111.1 |
| 8 | 786 | 87.9 |
| 16 | 451 | 76.6 |
| 32 | 244 | 70.8 |
| 64 | 197 | 44.0 |
| 128 | 238 | 18.2 |

Efficiency is calculated as: `(Reference runtime at 2 cores) / (Runtime at n cores) / (n/2) * 100`.  100% efficiency means linear scaling.

The table shows "superlinear scaling" (efficiency > 100%) from 2 to 4 cores, possibly due to CPU cache effects.  The 128-core test is slower than the 64-core test.  75% efficiency or higher is good; thus, 16 cores are recommended for this case.

The number of data points is flexible; at least five or six are recommended.  Stop if runtime increases with added cores.


### Weak Scaling

The problem size increases proportionally with the CPU core count.  Ideally, runtime remains constant.  An example:

| Cores | Problem Size | Run Time (s) | Efficiency (%) |
|---|---|---|---|
| 1 | 1000 | 3076 | - |
| 4 | 4000 | 3078 | 99.9 |
| 12 | 12,000 | 3107 | 99.0 |
| 48 | 48,000 | 3287 | 93.6 |
| 128 | 128,000 | 3966 | 77.6 |

Efficiency is calculated as: `(Reference runtime at 1 core) / (Runtime at n cores) * 100`.

Weak scaling is often better for memory-bound applications with nearest-neighbor communication.  Applications with significant nonlocal communication (e.g., Fast Fourier Transform) may show poor weak scaling.


[1] Wikipedia, "Super-linear speedup": https://en.wikipedia.org/wiki/Speedup#Super-linear_speedup

[2] Wikipedia, "Fast Fourier transform": https://en.wikipedia.org/wiki/Fast_Fourier_transform

Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Scalability&oldid=67674"
