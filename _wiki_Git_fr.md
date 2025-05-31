# Git

This page is a translated version of the page Git and the translation is 100% complete.

Other languages: English, français


## Description

Git is a fast and secure distributed tool for source code management (website, [gitscm.org](gitscm.org)). The application was created for the Linux project by Linus Torvalds and is maintained by Junio Hamano. Git development takes place on the [kernel.org](kernel.org) platform.


## Operating Principle

Unlike older source code management tools, Git works in a decentralized mode, and developers do not depend on a central repository to archive modifications. Each Git repository contains the complete working tree history. Each Git object (modification or changeset, file, directory) is a leaf of a multi-branch tree. Project development with Git is based on a model where a branch corresponds to a feature. Several iterations of the feature can be archived before it is merged with the common trunk. For details on the branching development model, see [A successful Git branching model](A successful Git branching model).

A particularly interesting technique is cherry-picking, which consists of taking part of a branch to merge it with another.


## Usage

As a general rule, a developer should be able to:

* Clone or create the repository;
* Make the modifications;
* Commit the modifications;
* Propagate the modifications to the original repository.

Since Git is decentralized, there is not necessarily an authoritative repository.


## Command Summary

### Basic Commands

| Command     | Description                                      |
|-------------|--------------------------------------------------|
| `git config` | Configure Git                                     |
| `git init`   | Create a new repository                           |
| `git clone`  | Clone an existing repository                      |
| `git add`    | Add a file or directory to the index              |
| `git rm`     | Remove a file or directory from the index         |
| `git commit` | Commit the changes in a repository               |
| `git push`   | Propagate committed changes to another repository |
| `git pull`   | Retrieve data from another repository and apply ('merge') the changes to your repository |
| `git fetch`  | Retrieve changes from a different repository without applying them to your repository |
| `git merge`  | Merge changes                                    |


### Commands to See Changes

| Command     | Description                               |
|-------------|-------------------------------------------|
| `git blame`  | Know the origin of each modification      |
| `git log`   | Get the history of records                |
| `git diff`  | See the differences between two versions   |
| `git status` | Display the status of files               |
| `git show`  | Display various types of Git objects       |
| `git cat-file` | Get the content, type, or size of objects |


### Commands Related to Branches, Tags, and Remote Repositories

| Command      | Description                             |
|--------------|-----------------------------------------|
| `git branch`  | Manage development branches              |
| `git tag`    | Manage version tags                      |
| `git remote` | Manage remote repositories                |
| `git checkout` | Check out a branch or path              |
| `git reset`   | Change the head                          |


### Commands Related to Patches

| Command          | Description                |
|-----------------|----------------------------|
| `git format-patch` | Create a patch             |
| `git am`         | Apply a patch              |
| `git send-email`  | Send a patch               |


### Other Commands

| Command     | Description                       |
|-------------|-----------------------------------|
| `git bisect` | Quickly search for a problem     |
| `git gc`     | Clean the repository              |
| `git rebase` | Linearize the history             |
| `git grep`   | Search the content                |


## Creating or Cloning a Repository

The first step is usually to create your own repository or clone an existing one.

To create a repository:

```bash
[name@server ~]$ git init my-project
```

To clone a repository:

```bash
[name@server ~]$ git clone git://github.com/git/git.git
```


## Committing and Saving a Modification

When the repository is ready, change directories and edit the file.

```bash
[name@server ~]$ cd my-project
[name@server ~]$ nano file.txt
```

When the work is finished, add the file:

```bash
[name@server ~]$ git add file.txt
```

and commit the modification:

```bash
[name@server ~]$ git commit
```

It is now possible to propagate the modifications to the original repository with:

```bash
[name@server ~]$ git push origin main
```

In this last command, `origin` is the remote repository and `main` is the current branch that will be propagated.  With older Git repositories, you may need to use `git push origin master`.


## Git Repository Hosting

GitHub and Bitbucket are the two main Git hosting services. They are both available for commercial projects and free projects.

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Git/fr&oldid=117254](https://docs.alliancecan.ca/mediawiki/index.php?title=Git/fr&oldid=117254)"
