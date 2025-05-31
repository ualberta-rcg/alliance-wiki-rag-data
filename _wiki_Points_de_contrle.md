# Checkpoints

Running a program sometimes takes longer than allowed by the submission systems on the clusters.  The execution of a long program is also subject to system uncertainties. A program with a short execution time can easily be restarted. However, when the program execution becomes very long, it is preferable to create checkpoints to minimize the chances of losing several weeks of computation. These will allow the program to be restarted later.

## Creating and Loading a Checkpoint

The creation and loading of a checkpoint may already be implemented in an application you are using.  Simply use this functionality and consult the documentation as needed.

However, if you have access to the application's source code and/or are the author, you can implement the creation and loading of checkpoints. Basically:

*   The creation of a checkpoint file is done periodically. Periods of 2 to 24 hours are suggested.
*   During the writing of the file, it is important to keep in mind that the computation task can be interrupted at any time, for any technical reason. Therefore:
    *   It is preferable not to overwrite the previous checkpoint when creating the new one.
    *   Writing can be made atomic by performing an operation that confirms the end of the checkpoint writing. For example, the file can initially be named according to the date and time, and finally, a symbolic link "last-version" can be created to the new checkpoint file with a unique name. Another more advanced method: a second file containing a hash sum of the checkpoint can be created, thus allowing validation of the checkpoint's integrity upon its eventual loading.
    *   Once the atomic write is complete, you can decide whether or not to delete old checkpoints.

## Resubmitting a Task for a Long Calculation

If a long calculation is expected to be broken down into several Slurm tasks, the two recommended methods are:

*   Using Slurm job arrays;
*   Resubmission from the end of the script.
