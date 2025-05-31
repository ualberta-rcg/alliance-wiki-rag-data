# Fortran

Fortran is a compiled language available on Compute Canada computers where the `gfortran` and `ifort` compilers are installed. In general, compiled languages offer better performance; therefore, we encourage you to write your programs in Fortran, C, or C++.

## Useful Compilation Options

Most modern Fortran compilers offer useful options for debugging:

*   `-fcheck=all` for the `gfortran` compiler and `-check` for the `ifort` compiler check array bounds and report untargeted pointers and uninitialized variables;
*   `-fpe0` (ifort) interrupts the application in floating-point cases (division by zero or square root of a negative number) rather than simply generating NaN (not a number) and letting the application continue;
*   during testing, use `-O0` to disable optimizations and `-g` to add debugging symbols.

## Numerical Linear Algebra

From Fortran 90 onwards, new functions are available for handling basic operations:

*   `matmul` and `dot_product` for matrix and vector multiplications;
*   `transpose` for matrix transposition.

Always use these functions or the provided BLAS/LAPACK libraries and never try to create your own methods unless for learning purposes. The BLAS routine for matrix multiplication can be 100 times faster than the primary algorithm with three nested loops.

## Segmentation Faults

A frequently observed error with a Fortran executable comes from interface problems. These problems occur when a pointer, a dynamically allocated array, or a function pointer is passed as an argument to a subroutine. There is no problem during compilation; however, during execution, you will get a message like this:

```
forrtl: severe (174): SIGSEGV, segmentation fault occurred
```

To correct the problem, you must ensure that the subroutine interface is explicitly defined. This can be done in Fortran with the `INTERFACE` command. Thus, the compiler will be able to build the interface, and the segmentation faults will be resolved.

In the case where the argument is an allocatable array, replace the following code:

**File: `error_allocate.f90`**

```fortran
Program Eigenvalue
implicit none
integer :: ierr
integer :: ntot
real, dimension(:,:), pointer :: matrix
read (5,*) ntot
ierr = genmat(ntot, matrix)
call Compute_Eigenvalue(ntot, matrix)
deallocate(matrix)
end
```

with this code:

**File: `interface_allocate.f90`**

```fortran
Program Eigenvalue
implicit none
integer :: ierr
integer :: ntot
real, dimension(:,:), pointer :: matrix
interface
function genmat(ntot, matrix)
implicit none
integer :: genmat
integer, intent(in) :: ntot
real, dimension(:,:), pointer :: matrix
end function genmat
end interface
read (5,*) ntot
ierr = genmat(ntot, matrix)
call Compute_Eigenvalue(ntot, matrix)
deallocate(matrix)
end
```

The principle is the same in the case where the argument is a function pointer. Consider, for example, the following code:

**File: `error_pointer.f90`**

```fortran
Program AreaUnderTheCurve
implicit none
real, parameter :: boundInf = 0.
real, parameter :: boundSup = 1.
real :: area
real, external :: computeIntegral
real, external :: FunctionToIntegrate
area = computeIntegral(FunctionToIntegrate, boundInf, boundSup)
end

function FunctionToIntegrate(x)
implicit none
real :: FunctionToIntegrate
real, intent(in) :: x
FunctionToIntegrate = x
end function FunctionToIntegrate

function computeIntegral(func, boundInf, boundSup)
implicit none
real, external :: func
real, intent(in) :: boundInf, boundSup
...
```

To avoid a segmentation fault, replace the previous code with the following:

**File: `interface_pointer.f90`**

```fortran
Program Eigenvalue
implicit none
real, parameter :: boundInf = 0.
real, parameter :: boundSup = 1.
real :: area
real, external :: computeIntegral
interface
function FunctionToIntegrate(x)
implicit none
real :: FunctionToIntegrate
real, intent(in) :: x
end function FunctionToIntegrate
end interface
area = computeIntegral(FunctionToIntegrate, boundInf, boundSup)
end

function FunctionToIntegrate(x)
implicit none
real :: FunctionToIntegrate
real, intent(in) :: x
FunctionToIntegrate = x
end function FunctionToIntegrate

function computeIntegral(func, boundInf, boundSup)
implicit none
real, intent(in) :: boundInf, boundSup
interface
function func(x)
implicit none
real :: func
real, intent(in) :: x
end function func
end interface
...
```


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Fortran/fr&oldid=68947](https://docs.alliancecan.ca/mediawiki/index.php?title=Fortran/fr&oldid=68947)"
