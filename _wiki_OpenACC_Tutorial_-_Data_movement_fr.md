# OpenACC Tutorial - Data Movement

This page is a translated version of the page [OpenACC Tutorial - Data movement](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Data_movement&oldid=14589) and the translation is 100% complete.

**Learning Objectives:**

* Understand the principles of locality and data movement.
* Know the difference between structured and unstructured data.
* Know how to perform explicit data transfer.
* Know how to compile and run OpenACC code with movement directives.


## 1 Explicit Data Management

### Data Management with Unified Memory

We used CUDA Unified Memory to simplify the first steps of accelerating our code. While simpler, this code is not portable:

* For PGI only, flag `-ta=tesla:managed`
* For NVIDIA only, CUDA Unified Memory

Explicit data management makes the code portable and can improve performance.


## 2 Structured Data Regions

The `data` directive delimits the code region where GPU arrays remain on the GPU and are shared by all kernels in the region.

Here is an example of how a structured data region is defined:

```c++
#pragma acc data
{
#pragma acc parallel loop ...
#pragma acc parallel loop
...
}
```

Another example:

```fortran
!$acc data
!$acc parallel loop
...
!$acc parallel loop
...
!$acc end data
```

**Data Locality:** Arrays inside the data region remain on the GPU until the end of the region.


## 3 Unstructured Data

In some cases, delimiting a region does not allow the use of normal data regions, for example when using constructors or destructors.


### 3.1 Directives

In these cases, unstructured data directives are used:

* `enter data`, defines the beginning of the lifetime of unstructured data. Clauses: `copyin(list)`, `create(list)`
* `exit data`, defines the end of the lifetime of unstructured data. Clauses: `copyout(list)`, `delete(list)`

Here is an example:

```c++
#pragma acc enter data copyin(a)
...
#pragma acc exit data delete(a)
```


### 3.2 C++ Classes

What is the advantage of unstructured data clauses? They allow the use of OpenACC in C++ classes.  Furthermore, these clauses can be used when data is allocated and initialized in a different part of the code than where the data is freed, for example in Fortran modules.

```c++
class Matrix {
Matrix(int n) {
len = n;
v = new double[len];
#pragma acc enter data create(v[0:len])
}
~Matrix() {
#pragma acc exit data delete(v[0:len])
};
```


### 3.3 Clauses of the `data` Directive

* `copyin(list)`, to allocate GPU memory and copy data from the host memory to the GPU at the entrance of the region.
* `copyout(list)`, to allocate GPU memory and copy data to the host memory at the exit of the region.
* `copy(list)`, to allocate GPU memory and copy data from the host memory to the GPU at the entrance of the region and copy data to the host memory at the exit of the region (structured data only).
* `create(list)`, to allocate GPU memory without copying.
* `delete(list)`, to deallocate GPU memory without copying (unstructured data only).
* `present(list)`, the GPU already contains data from another region.


### 3.4 Array Format

The compiler cannot always determine the size of an array; therefore, the size and format must be specified. Here is an example in C:

```c++
#pragma acc data copyin(a[0:nelem]) copyout(b[s/4:3*s/4])
```

And an example in Fortran:

```fortran
!$acc data copyin(a(1:end)) copyout(b(s/4:3*s/4))
```


## 4 Explicit Data Movement

### 4.1 Copying into the Matrix

In this example, we start by allocating and initializing the matrix. The matrix is then copied into memory. The copy is done in two steps:

1. Copy the matrix structure.
2. Copy the members of the matrix.

```c++
void allocate_3d_poisson_matrix(matrix &A, int N) {
int num_rows = (N + 1) * (N + 1) * (N + 1);
int nnz = 27 * num_rows;
A.num_rows = num_rows;
A.row_offsets = (unsigned int*) malloc((num_rows + 1) * sizeof(unsigned int));
A.cols = (unsigned int*) malloc(nnz * sizeof(unsigned int));
A.coefs = (double*) malloc(nnz * sizeof(double));
// Initialize Matrix
A.row_offsets[num_rows] = nnz;
A.nnz = nnz;
#pragma acc enter data copyin(A)
#pragma acc enter data copyin(A.row_offsets[:num_rows+1],A.cols[:nnz],A.coefs[:nnz])
}
```


### 4.2 Deleting the Matrix

To free the memory, you must first exit the matrix and then issue the `free` command. This is done in two steps, but in reverse order:

1. Delete the members.
2. Delete the structure.

```c++
void free_matrix(matrix &A) {
unsigned int *row_offsets = A.row_offsets;
unsigned int *cols = A.cols;
double *coefs = A.coefs;
#pragma acc exit data delete(A.row_offsets,A.cols,A.coefs)
#pragma acc exit data delete(A)
free(row_offsets);
free(cols);
free(coefs);
}
```


### 4.3 The `present` Clause

For high-level management, you must tell the compiler that the data is already in memory. However, the declaration of local variables should be done inside the function in which they are used.

```c++
function main(int argc, char **argv) {
#pragma acc data copy(A) {
laplace2D(A, n, m);
}
}
...
function laplace2D(double[N][M] A, n, m){
#pragma acc data present(A[n][m]) create(Anew)
while (err > tol && iter < iter_max) {
err = 0.0;
...
}
}
```

Use `present` whenever possible.  Critical elements for good performance are high-level management and the use of the `present` clause.

In the next example, the computation region in the code contains the information that tells the compiler that the data is already present.

```c++
#pragma acc kernels \
present(row_offsets,cols,Acoefs,xcoefs,ycoefs)
{
for (int i = 0; i < num_rows; i++) {
double sum = 0;
int row_start = row_offsets[i];
int row_end = row_offsets[i + 1];
for (int j = row_start; j < row_end; j++) {
unsigned int Acol = cols[j];
double Acoef = Acoefs[j];
double xcoef = xcoefs[Acol];
sum += Acoef * xcoef;
}
ycoefs[i] = sum;
}
}
```


### 4.4 Compiling and Running with Explicit Memory Management

To make a new build without managed memory, replace `-ta=tesla:managed` with `-ta=tesla` in the Makefile.


### 4.5 The `update` Directive

This directive allows you to update an array or part of an array.

```fortran
do_something_on_device()
!$acc update self(a) // Copy "a" from GPU to CPU
do_something_on_host()
!$acc update device(a) // Copy "a" from CPU to GPU
```

In this other example, we first modify a vector in the host CPU memory, then copy it to the GPU memory.

```c++
void initialize_vector(vector &v, double val) {
for (int i = 0; i < v.n; i++)
v.coefs[i] = val;
// Updating the vector on the CPU
#pragma acc update device(v.coefs[:v.n])
// Updating the vector on the GPU
}
```


### 4.6 Developing and Running without Managed Memory

Here we see the performance of the code with and without managed memory.

**Benchmark with and without managed memory:** In this example, tests were performed with and without the `-ta=tesla:managed` option.

**Other results:** The results show that some tests with managed memory improve speed; this is probably due to pinned memory. In general, it seems that locality works: when most operations are performed on the GPU and the data remains there for a long time, data movement does not have a major impact on performance.


## Exercise: Adding Directives

Use the `kernels` or `parallel loop` directives to obtain explicit data management. The `step2.*` directories on Github contain the solution. Modify the compiler flags to `-ta=tesla` (unmanaged). Check if the results and performance are the same as before.


[Next page: Loop Optimization](link_to_next_page)

[Return to the beginning of the tutorial](link_to_beginning)
