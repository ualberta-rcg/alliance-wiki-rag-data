# Git

## Description

Git is a distributed, fast, and secure source code management tool. The official Git website is [gitscm.org](gitscm.org). The Git software was initially created by Linus Torvalds for the Linux project, and the current maintainer is Junio Hamano. The development of Git itself is performed on the [kernel.org](kernel.org) platform.

## Operating Principle

Contrary to older source code management tools, Git works in a distributed way. This means that developers do not depend on a central repository to commit their changes. Each Git repository contains the full history of the project. Each Git object (changeset, file, directory) is the leaf of a tree with multiple branches. Developing a project with Git is based on a model in which one branch corresponds to one feature. Many revisions of the feature may be archived before the branch gets merged with the main trunk. For a detailed explanation of branch development, we recommend reading [this page](link_to_page_needed).

One especially interesting technique is cherry-picking, which is essentially taking part of a branch and merging it with another one.

## Basic Usage

Generally, a project developer must be able to:

*   clone or create the repository;
*   make changes;
*   commit changes;
*   push changes toward the original repository.

Since Git is distributed, there may not be an authoritative repository.

## Summary of Commands

### Basic Commands

| Command     | Description                     |
|-------------|---------------------------------|
| `git config` | Configures git                   |
| `git init`   | Creates a new repository         |
| `git clone`  | Clones an existing repository    |
| `git add`    | Adds a file or directory to a repository |
| `git rm`     | Deletes a file or directory from the repository |
| `git commit` | Commits changes to the repository |
| `git push`   | Pushes committed changes to a different repository |
| `git pull`   | Pulls changes from a different repository and merges them with your own repository |
| `git fetch`  | Fetches changes from a different repository without merging them to yours |
| `git merge`  | Merges changes to the repository |

### Commands to Explore Changes

| Command     | Description                     |
|-------------|---------------------------------|
| `git blame`  | Gives the origin of each change |
| `git log`   | Displays changes history        |
| `git diff`  | Compares two versions           |
| `git status` | Displays status of the current files |
| `git show`  | Displays various git objects     |
| `git cat-file` | Displays the content, type or size of objects |

### Commands for Branches, Tags, and Remote Repositories

| Command      | Description                       |
|--------------|-----------------------------------|
| `git branch`  | Manages development branches      |
| `git tag`    | Manages version tags              |
| `git remote` | Manages remote repositories       |
| `git checkout` | Checks out a branch or a path     |
| `git reset`  | Changes the head of a branch      |

### Commands for Patches

| Command          | Description                |
|-----------------|----------------------------|
| `git format-patch` | Creates a patch            |
| `git am`         | Applies a patch            |
| `git send-email` | Sends a patch by email     |

### Other Commands

| Command     | Description                       |
|-------------|-----------------------------------|
| `git bisect` | Used to diagnose problems        |
| `git gc`     | Collects garbage objects         |
| `git rebase` | Rebases history of the repository |
| `git grep`   | Searches for content            |


## Creating or Cloning a Repository

The first step is usually to create your own repository, or to clone an existing one.

To create a repository:

```bash
[name@server ~]$ git init my-project
```

To clone a repository:

```bash
[name@server ~]$ git clone git://github.com/git/git.git
```

## Committing a Change

When the repository is ready, you change directory and edit the file.

```bash
[name@server ~]$ cd my-project
[name@server ~]$ nano file.txt
```

When work is done, you add the file

```bash
[name@server ~]$ git add file.txt
```

and commit the change

```bash
[name@server ~]$ git commit
```

It is then possible to push changes to the origin repository with

```bash
[name@server ~]$ git push origin main
```

In the above command, `origin` is the remote repository and `main` is the current branch that will be pushed. You might have to use `git push origin master` for older Git repositories.

## Hosting Git Repositories

GitHub and Bitbucket are two of the main Git repository hosting services. They are both available for commercial projects as well as free projects.
