# Gaussian Error Messages

This page is a translated version of the page [Gaussian error messages](https://gaussian.com/gaussian-error-messages/) and the translation is 100% complete.

Other languages: [English](https://gaussian.com/gaussian-error-messages/), [français](https://docs.alliancecan.ca/mediawiki/index.php?title=Gaussian_error_messages/fr&oldid=117775)

This information was originally compiled by Professor Cory C. Pye.


## Normal termination of Gaussian

Under normal execution conditions, Gaussian terminates as follows:

```
Job cpu time:       0 days  0 hours 26 minutes 18.3 seconds.
 Elapsed time:       0 days  0 hours  6 minutes 43.3 seconds.
 Normal termination of Gaussian 16 at Tue Nov 14 15:31:56 2017.
```

If the job fails without producing an error message and the output report otherwise seems normal, this may indicate that:

* Your quota has been exceeded (see [Quotas and policies](link-to-quotas-and-policies)).
* The execution time was longer than the requested time (see [the scheduler](link-to-scheduler)) (`--time=HH:MM:SS`).
* Insufficient memory was requested (see [Job monitoring](link-to-job-monitoring)). Or
* Your job produces data whose size exceeds the capacity of the local disk of the computing nodes.


## Erroneous Write

**Description**

Towards the end of the output file, a variant of the following may be read:

```
Erroneous write. write 122880 instead of 4239360.
fd = 3
Erroneous write. write 122880 instead of 4239360.
fd = 3
writwa
writwa: File exists
```

or

```
Erroneous write. write -1 instead of 3648000. 

fd = 4
writwa
writwa: No space left on device
```

or

```
Erroneous write during file extend. write -1 instead of 8192
Probably out of disk space.
Write error in NtrExt1
```

**Cause**

This error usually occurs when disk space is exhausted. The error may occur if you have exceeded your quota, if the disk is full, or, in the rarer case where a network drive is unavailable due to a communication interruption.

**Solution**

Check your quota with `quota`; if necessary, delete unnecessary files.  Your job may be too large to run on the hardware; try reducing the basis set.


## Link 9999

**Description**

At the end of the output file, a variant of the following may be read:

```
Error termination request processed by link 9999.
 Error termination via Lnk1e in /disc30/g98/l9999.exe.
```

A few pages earlier, a variant of the following may be read:

```
Maximum Force            0.020301     0.000450     NO 
 RMS     Force            0.007068     0.000300     NO 
 Maximum Displacement     0.078972     0.001800     NO 
 RMS     Displacement     0.023716     0.001200     NO 
 Predicted change in Energy=-3.132299D-05
 Optimization stopped.
    -- Number of steps exceeded,  NStep=  34
    -- Flag reset to prevent archiving.
                       ----------------------------
                       ! Non-Optimized Parameters !
                       ! (Angstroms and Degrees)  !
```

**Cause**

The job terminated abnormally due to an internal reason within the application. The most frequent cause is the non-convergence of a geometric optimization.

**Solution**

Optimize the starting structure at a lower theoretical level. However, if the viewer shows convergence as desired, restart the optimization from the last step using, for example, `geom=allcheck` in the line specifying the direction. It might be a good idea to use `opt=CalcFC` if it is not too expensive, which is probably the case at HF or DFT levels.

Use a richer Hessian matrix of force constants. This would generally be the case if the force constants vary greatly from one level to another, or if there is a significant geometric change in the optimization. A series of related jobs can be run with `--Link1--`. If you have a previous job, `Opt=ReadFC` will usually give good results, but occasionally also `Opt=CalcFC` and more rarely `Opt=CalcAll`. In these cases, the forces are often convergent, but the increments are not, giving:

```
Item               Value     Threshold  Converged?
 Maximum Force            0.000401     0.000450     YES
 RMS     Force            0.000178     0.000300     YES
 Maximum Displacement     0.010503     0.001800     NO 
 RMS     Displacement     0.003163     0.001200     NO
```

Occasionally, the problem comes from the coordinate system, especially with a Z-matrix where it is easy to make bad choices. In several cases, three of the four atoms used to define the dihedral angle may become collinear, i.e., the angle is close to 0 or 180 degrees, which can cause the algorithm to fail. You can either modify your Z-matrix or use the redundant internal coordinates defined by default.

As a last resort, replace the default optimization method with another type of method, for example `opt=ef` (for less than 50 variables) or `opt=gdiis` (for flexible molecules).


## angle Alpha is outside the valid range of 0 to 180

**Description**

At the end of the output file, a variant of the following may be read:

```
------------------------------------------------------------------------
 Error termination via Lnk1e in /disc30/g98/l716.exe.
```

The lines above will be a z-matrix, above which will contain lines such as:

```
Error on Z-matrix card number    9
 angle Alpha is outside the valid range of 0 to 180.
 Conversion from Z-matrix to cartesian coordinates failed:
 ------------------------------------------------------------------------
                         Z-MATRIX (ANGSTROMS AND DEGREES)
 CD Cent Atom  N1     Length/X     N2    Alpha/Y     N3     Beta/Z      J
 ------------------------------------------------------------------------
...
  9   9  H     8   0.962154(  8)   1   -1.879( 16)   2    0.000( 23)   0
...
```

**Cause**

The job terminated abnormally because one of the X angles of the Z-matrix was found to be outside the allowed limits of 0 < X < 180.

**Solution**

This can occur in the case of significant geometric modifications in a molecule, especially when it is composed of interacting fragments. Redefine the Z-matrix or use another coordinate system.


## Reading basis center

**Description**

At the end of the output file, a variant of the following may be read:

```
End of file reading basis center.
 Error termination via Lnk1e in /disc30/g98/l301.exe.
 Job cpu time:  0 days  0 hours  0 minutes  1.9 seconds.
 File lengths (MBytes):  RWF=   11 Int=    0 D2E=    0 Chk=   10 Scr=    1
```

**Cause**

This is an input error. You want to read a general basis, but you failed to specify it.

**Solution**

Enter the basis set in question, or remove `gen` from the route line and specify an internal basis set.


## Operation on file out of range

**Description**

At the end of the output file, a variant of the following may be read:

```
Error termination in NtrErr:
 NtrErr Called from FileIO.
```

Preceded by:

```
Operation on file out of range.
FileIO: IOper= 2 IFilNo(1)=-19999 Len=     1829888 IPos=  -900525056 Q=       4352094416

 dumping /fiocom/, unit = 1 NFiles =   109 SizExt =    524288 WInBlk =      1024
                   defal = T LstWrd =  7437256704 FType=2 FMxFil=10000
```

And followed by several numbers.

**Solution**

You are using `Opt=ReadFC`, `guess=read`, or `geom=allcheck/modify` to obtain from the checkpoint file something that is not found because the calculation was not done or the information is absent from the checkpoint file because the previous job was not completed due to lack of time or disk space.

**Solution**

Resume calculations or enter the required information.


## End of file in GetChg

**Description**

At the end of the output file, a variant of the following may be read:

```
Symbolic Z-matrix:
 End of file in GetChg.
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=   11 Scr=    1
```

**Cause**

You failed to enter the charge/multiplicity line in input or you wanted to use charge/multiplicity from the checkpoint file, but you omitted `geom=allcheck` in the route section.

**Solution**

Enter the charge/multiplicity line or add `geom=allcheck` in the route section.


## Change in point group or standard orientation

**Description**

At the end of the output file, a variant of the following may be read:

```
Stoichiometry    CdH14O7(2+)
 Framework group  C2[C2(CdO),X(H14O6)]
 Deg. of freedom   30
 Full point group                 C2      NOp   2
 Omega: Change in point group or standard orientation.

 Error termination via Lnk1e in /disc30/g98/l202.exe.
 Job cpu time:  0 days  3 hours 35 minutes 40.8 seconds.
 File lengths (MBytes):  RWF=   58 Int=    0 D2E=    0 Chk=   19 Scr=    1
```

**Cause**

The standard orientation or point group of the molecule was modified during the optimization. In the latter case, a visualization program will show a sudden reversal of the structure, generally 180 degrees. This error has occurred less since the Gaussian 03 version.

**Solution**

Your Z-matrix may be poorly defined if you go from one point group to a subgroup of that point group (e.g., from C2v to C2, Cs, or C1).

If the point group is correct, it may be that the symmetry of the starting structure is too high and needs to be reduced.

In some rare cases, the point group is incorrect or the symmetry is too high; reformulate the Z-matrix with more symmetry.

If symmetry is not important, disable it.


## Unrecognized atomic symbol

**Description**

At the end of the output file, a variant of the following may be read:

```
General basis read from cards:  (6D, 7F)
 Unrecognized atomic symbol ic2 

 Error termination via Lnk1e in /disc30/g98/l301.exe.
 Job cpu time:  0 days  0 hours  0 minutes  1.6 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=   12 Scr=    1
```

**Cause**

The reading is done in a general basis, but the specified atom (`ic2` in this example) does not correspond to any standard atomic symbol. This can also occur in a linked job if, in a previous step, default coordinates were used and erased the Z-matrix while you then attempt to modify it with `geom=modify`. The variable section is ignored, but the application may attempt to interpret it as part of the basis.

**Solution**

Enter the correct atomic symbol.


## Convergence failure -- run terminated

**Description**

At the end of the output file, a variant of the following may be read:

```
>>>>>>>>>> Convergence criterion not met.
 SCF Done:  E(RHF) =  -2131.95693715     A.U. after  257 cycles
             Convg  =    0.8831D-03             -V/T =  2.0048
             S**2   =   0.0000
 Convergence failure -- run terminated.
 Error termination via Lnk1e in /disc30/g98/l502.exe.
 Job cpu time:  0 days  0 hours  5 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=   15 Int=    0 D2E=    0 Chk=    8 Scr=    1
```

or

```
>>>>>>>>>> Convergence criterion not met.
 SCF Done:  E(UHF) =  -918.564956094     A.U. after   65 cycles
             Convg  =    0.4502D-04             -V/T =  2.0002
             S**2   =   0.8616
 Annihilation of the first spin contaminant:
 S**2 before annihilation     0.8616,   after     0.7531
 Convergence failure -- run terminated.
 Error termination via Lnk1e in /disc30/g98/l502.exe.
 Job cpu time:  0 days  0 hours  3 minutes 56.2 seconds.
 File lengths (MBytes):  RWF=   11 Int=    0 D2E=    0 Chk=    8 Scr=    1
```

**Cause**

The SCF (self-consistent field) procedure did not converge.

**Solution**

This can occur when the molecular orbitals have a poor `guess=read`. Try to obtain a better `guess=read` by running an SCF procedure with the same starting structure, but with a lower theoretical level, for example HF/STO-3G.

If this does not work, use a different convergence procedure, such as `SCF=QC` or `SCF=XQC`.

In some cases, a weakness in the geometry can prevent convergence if one of the bonds is either too long or too short. The problem can be solved by modifying the initial geometry.

The error can also result from a step in the optimization procedure that was poorly performed.

Submit the job again using the penultimate geometry (or an earlier geometry) and a new evaluation of the Hessian matrix.


## FOPT requested but NVar= XX while NDOF= YY

**Description**

At the end of the output file, a variant of the following may be read:

```
FOPT requested but NVar= 29 while NDOF= 15.
 Error termination via Lnk1e in /disc30/g98/l202.exe.
 Job cpu time:  0 days  0 hours  0 minutes  1.3 seconds.
 File lengths (MBytes):  RWF=   11 Int=    0 D2E=    0 Chk=    1 Scr=    1
```

**Cause**

You requested a full optimization (FOpt), including a check of the correct number of variables. The check reported an error.

**Solution**

If NDOF is smaller than NVar, the molecule is run with a symmetry lower than it is. Increase the symmetry.

If NVar is smaller than NDOF, your Z-matrix has too many constraints for the symmetry in question.

The check can be bypassed by using Opt instead of FOpt; however, this is not recommended.


## Unable to project read-in occupied orbitals

**Description**

At the end of the output file, a variant of the following may be read:

```
Initial guess read from the checkpoint file:
 BiAq7_3+_C2.chk
 Unable to project full set of read-in orbitals.
 Projecting just the  36 occupied ones.
 Unable to project read-in occupied orbitals.
 Error termination via Lnk1e in /disc30/g98/l401.exe.
 Job cpu time:  0 days  0 hours  0 minutes 29.5 seconds.
 File lengths (MBytes):  RWF=   18 Int=    0 D2E=    0 Chk=   17 Scr=    1
```

**Cause**

You are reading the guess from a molecular orbital that comes from the checkpoint file, but the projection of the old basis to the new one did not work. This can occur when some pseudopotential bases (CEP-121G*) are used with polarization functions while one of these functions does not exist. In some cases, Gaussian uses temporary polarization functions with zero exponent.

**Solution**

Use CEP-121G instead of CEP-121G*; they are the same for several elements. You can also bypass the problem by avoiding using the guess.


## KLT.ge.NIJTC in GetRSB

**Description**

At the end of the output file, a variant of the following may be read:

```
(rs|ai) integrals will be sorted in core.
 KLT.ge.NIJTC in GetRSB.
 Error termination via Lnk1e in /disc30/g98/l906.exe.
 Job cpu time:  0 days  0 hours  0 minutes 32.7 seconds.
 File lengths (MBytes):  RWF=  514 Int=    0 D2E=    0 Chk=   10 Scr=    1
```

**Cause**

The MP2 calculation failed, perhaps due to the pseudopotential problem mentioned in relation to the previous error message.

**Solution**

Use CEP-121G instead of CEP-121G*; they are the same for several elements.


## Symbol XXX not found in Z-matrix

**Description**

At the end of the output file, a variant of the following may be read:

```
Symbol "H3NNN" not found in Z-matrix.
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=   14 Scr=    1
```

**Cause**

You entered a variable name (here H3NNN) that is not found in the Z-matrix.

**Solution**

Enter the correct variable name or add it to the Z-matrix.


## Variable X has invalid number of steps

**Description**

At the end of the output file, a variant of the following may be read:

```
Scan the potential surface.
 Variable   Value     No. Steps Step-Size
 -------- ----------- --------- ---------
 Variable  1 has invalid number of steps      -1.
 Error termination via Lnk1e in /disc30/g98/l108.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.7 seconds.
 File lengths (MBytes):  RWF=   11 Int=    0 D2E=    0 Chk=   13 Scr=    1
```

**Cause**

This is an input error. You are trying to generate a rigid scan of the potential energy and there are probably two blank lines instead of one between the Z-matrix and the variables.

**Solution**

Remove the blank line.


## Problem with the distance matrix

**Description**

At the end of the output file, a variant of the following may be read:

```
Problem with the distance matrix.
 Error termination via Lnk1e in /disc30/g98/l202.exe.
 Job cpu time:  0 days  9 hours 11 minutes 14.3 seconds.
 File lengths (MBytes):  RWF=  634 Int=    0 D2E=    0 Chk=   10 Scr=    1
```

**Cause**

This may be an input error. At least two atoms are too close to each other in this list. This is sometimes a programming error, especially when one of the distances is NaN (not a number). This can occur in the optimization of diatomic molecules when the initial distance is too large.

**Solution**

Check the variables and the Z-matrix of the atoms in question to see if some atoms are too close together. This could be the result of the absence of a subtraction sign in a torsion angle for molecules with symmetric planes where the connected atoms do not coincide, i.e., the distance between them is zero.


## End of file in ZSymb

**Description**

At the end of the output file, a variant of the following may be read:

```
Symbolic Z-matrix:
 Charge =  0 Multiplicity = 1
 End of file in ZSymb.
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.6 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=    9 Scr=    1
```

**Cause**

This is an input error. The matrix is not found for one of these reasons:

* You may have omitted the blank line at the end of the geometry specifications.
* You wanted to get the Z-matrix and parameters from the checkpoint file, but you forgot to enter `geom=check`.

**Solution**

Add a blank line at the end or add `geom=check`.


## Linear search skipped for unknown reason

**Description**

At the end of the output file, a variant of the following may be read:

```
RFO could not converge Lambda in  999 iterations.
 Linear search skipped for unknown reason.
 Error termination via Lnk1e in /disc30/g98/l103.exe.
 Job cpu time:  0 days  7 hours  9 minutes 17.0 seconds.
 File lengths (MBytes):  RWF=   21 Int=    0 D2E=    0 Chk=    6 Scr=    1
```

**Cause**

The RFO (rational function optimization) did not work during a linear search. The Hessian matrix is probably no longer valid.

**Solution**

Restart the optimization with `Opt=CalcFC`.


## Variable index of 3000 on card XXX is out of range, NVar=XX

**Description**

At the end of the output file, a variant of the following may be read:

```
Variable index of 3000 on card  15 is out of range, NVar=  42.
 Termination in UpdVr1.
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=   11 Int=    0 D2E=    0 Chk=    8 Scr=    1
```

**Cause**

This is an input error. You forgot to add a variable to your Z-matrix list; in this example, the variable defines atom number 15.

**Solution**

Add the variable.


## Unknown center XXX

**Description**

At the end of the output file, a variant of the following may be read:

```
Unknown center X
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=    8 Scr=    1
```

**Cause**

This is an input error. You are trying to define an atom in a Z-matrix using another non-existent atom (X in this example).

**Solution**

Use the correct atom name.


## Determination of dummy atom variables in z-matrix conversion failed

**Description**

At the end of the output file, a variant of the following may be read:

```
Error termination request processed by link 9999.
 Error termination via Lnk1e in /disc30/g98/l9999.exe.
 Job cpu time:  0 days  1 hours 53 minutes 10.4 seconds.
 File lengths (MBytes):  RWF=   20 Int=    0 D2E=    0 Chk=   11 Scr=    1
```

And just before:

```
Determination of dummy atom variables in z-matrix conversion failed.
 Determination of dummy atom variables in z-matrix conversion failed.
 NNew=      6.03366976D+01 NOld=      5.07835896D+01 Diff= 9.55D+00
```

**Cause**

The conversion of redundant internal coordinates to Z-matrix coordinates failed due to dummy atoms. You will have to use Cartesian coordinates.

**Solution**

The geometric optimization converged, but Gaussian was unable to reconvert to the input Z-matrix.


## malloc failed

**Description**

At the end of the output file, the following is read:

```
malloc failed.: Resource temporarily unavailable
malloc failed.
```

**Cause**

This is not a Gaussian error per se. This indicates a lack of memory, perhaps because you requested too much memory on the `%mem` line.

**Solution**

Decrease the value of `%mem` or increase the amount of memory indicated in the job script with `--mem=`.


## Charge and multiplicity card seems defective

**Description**

At the end of the output file, a variant of the following may be read:

```
----
 -2 1
 ----
 Z-Matrix taken from the checkpoint file:
 oxalate_2-_Aq1_C2.chk
 Charge and multiplicity card seems defective:
 Charge is bogus.
  WANTED AN INTEGER AS INPUT.
  FOUND A STRING AS INPUT.
 CX      =  0.7995                                                              
                                                        
   ?
 Error termination via Lnk1e in /disc30/g98/l101.exe.
```

**Cause**

This is an input error. In the absence of a title line with `geom=modify`, the charge/multiplicity line is interpreted as being the title (-2 1 in this example) and the charge/multiplicity line is interpreted as being the list of variables.

**Solution**

Enter a title line.


## Attempt to redefine unrecognized symbol "XXXXX"

**Description of error**

At the end of the output file, a variant of the following may be read:

```
O2WXC  90. 
 Attempt to redefine unrecognized symbol "O2WXC".
 Error termination via Lnk1e in /disc30/g98/l101.exe.
 Job cpu time:  0 days  0 hours  0 minutes  0.5 seconds.
 File lengths (MBytes):  RWF=    6 Int=    0 D2E=    0 Chk=    8 Scr=    1
```

**Cause**

This is an input error. You are requesting `geom=modify`, but one of the variables you are trying to move is not found in the checkpoint file.

**Solution**

Enter the correct checkpoint file or the correct variable.


## Inconsistency #2 in MakNEB

**Description**

At the end of the output file, a variant of the following may be read:

```
Standard basis: 3-21G (6D, 7F)
 Inconsistency #2 in MakNEB.
 Error termination via Lnk1e in /disc30/g98/l301.exe.
 Job cpu time:  0 days  3 hours 46 minutes 57.4 seconds.
 File lengths (MBytes):  RWF=  245 Int=    0 D2E=    0 Chk=   11 Scr=    1
```

**Cause**

This is an input error. The point group has been modified and you have specified `iop(2/15=4,2/16=2,2/17=7)` so that the program does not crash.

**Solution**

Be very careful with `iop` or remove it.


## galloc: could not allocate memory

**Description**

In the output file, the following is read:

```
galloc: could not allocate memory
```

**Cause**

This is an allocation error due to lack of memory. Gaussian uses approximately 1GB more than `%mem`.

**Solution**

The value of `%mem` must be at least 1GB less than the value specified in the job script. Similarly, the value of `--mem` specified in the script must be at least 1GB greater than the amount specified by the `%mem` directive of the input file. The appropriate increment seems to depend on the type of job and the details in the input file; 1GB is a conservative empirically determined value.


## No such file or directory

**Description**

In the output file, a variant of the following may be read:

```
PGFIO/stdio: No such file or directory
PGFIO-F-/OPEN/unit=11/error code returned by host stdio - 2.
 File name = /home/johndoe/scratch/Gau-12345.inp
 In source file ml0.f, at line number 181
  0  0x42bb41
Error: segmentation violation, address not mapped to object
```

**Cause**

The file mentioned on the third line does not exist, possibly because the directory containing it does not exist. This can occur, for example, if you assign to `GAUSS_SCRDIR` a directory that does not exist.

**Solution**

Create the directory with `mkdir` or modify the definition of `GAUSS_SCRDIR` to use an existing directory.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Gaussian_error_messages/fr&oldid=117775](https://docs.alliancecan.ca/mediawiki/index.php?title=Gaussian_error_messages/fr&oldid=117775)"
