# Migration to the New Standard Environment

This page is a translated version of the page [Migration to the new standard environment](https://docs.alliancecan.ca/mediawiki/index.php?title=Migration_to_the_new_standard_environment&oldid=150127) and the translation is 100% complete.

Other languages:

* [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Migration_to_the_new_standard_environment&oldid=150127)
* français


## What are the differences between StdEnv/2023 and other standard environments?

Refer to the [Standard Software Environments](https://docs.alliancecan.ca/mediawiki/index.php?title=Standard_Software_Environments&oldid=150127) page.


## Can I change my default standard environment?

After April 1st, 2024, `StdEnv/2023` will be the default environment for all our clusters.  However, it is still possible to modify the `$HOME/.modulerc` file. For example, the following command will set your default environment to `StdEnv/2020`:

```bash
echo "module-version StdEnv/2020 default" >> $HOME/.modulerc
```

For this to take effect, you must log out and log back in.


## Do I need to reinstall/recompile my code when the standard environment is changed?

Yes. If you compile your own code or have installed R or Python packages, you must recompile or reinstall the packages with the new environment.


## How can I use an older environment?

If you have ongoing work and do not want to change the versions of the software you are currently using, add the command `module load StdEnv/2020` to your job scripts before loading other modules.


## Will older versions be deleted?

Older environments will remain available, as well as the software that depends on them. However, versions 2016.4 and 2018.3 are no longer supported and we recommend that you do not use them. Our team will only install software in the new 2023 environment.


## Is it possible to use modules from different environments together?

No, you will get unpredictable results and probably errors. In each task, you can explicitly load one or the other environment, but only one environment per task.


## Which environment should I use?

We recommend using `StdEnv/2023` for your new projects or if you want to use a newer version of a software. To do this, add the command `module load StdEnv/2023` to your job scripts. It is not necessary to remove this command to use `StdEnv/2023` after April 1st.


## Can I keep my current environment by loading modules in my .bashrc?

It is not recommended to load modules in your `.bashrc`. Instead, load the modules via scripts for your tasks.


## I only use cloud resources; does the environment change affect me?

No, this change only affects the use of the available software loaded via modules.


## I can no longer load a module that I used before the change

The new environment contains newer versions of most applications. To find out about these versions, run the command `module avail`. For example:

```bash
module avail gcc
```

This shows several versions of GCC compilers, which may be different from those in older environments.
