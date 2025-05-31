# Prolonging Terminal Sessions

This page is a translated version of the page [Prolonging terminal sessions](https://docs.alliancecan.ca/mediawiki/index.php?title=Prolonging_terminal_sessions&oldid=139817) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Prolonging_terminal_sessions&oldid=139817), français

To submit and monitor tasks, modify files, and perform many other operations, you will likely need to connect to a cluster via SSH.  Sometimes it's necessary to keep the connection active for several hours, or even several days. We describe some techniques to achieve this here.

## SSH Configuration

A simple solution to prolong a connection is to modify your SSH client configuration. On macOS and Linux, this configuration is located in `$HOME/.ssh/config`, while on Windows it's in `C:\Users\<username>\.ssh\config`. If the file doesn't exist, create it and add the following lines:

```
Host *
    ServerAliveInterval 240
```

This sends a keep-alive signal to the remote server (like an Alliance cluster) every 240 seconds (4 minutes), which should keep the connection alive even if it's inactive for a few hours.

## Terminal Multiplexer

The programs `tmux` and `screen` are examples of terminal multiplexers that allow you to completely detach a terminal session, which will remain active until you reattach to it. You can therefore disconnect from the cluster, close your workstation, or put it to sleep, and then resume work the next day by reattaching to the same session.

### Connection Node Dependency

Each of our clusters includes several connection nodes, and your `tmux` or `screen` session is launched on a particular node. To reattach to a session, you must use the same connection node as the one where you launched `tmux` or `screen`.  It can happen that a connection node is restarted, which eliminates the sessions located on that node; in such a case, your sessions and all associated processes will be lost.

### tmux

`tmux` is a terminal multiplexer that allows multiple virtual sessions within a single terminal session. You can therefore disconnect from an SSH session without affecting the processes.

For an introduction to `tmux`:

* [The Tao of tmux](https://tmux.github.io/)
* [Getting Started With TMUX](https://www.youtube.com/watch?v=s_o-k-p-1qE) (24-minute video)
* [Turbo boost your interactive experience on the cluster with tmux](https://www.youtube.com/watch?v=o_P0t7tR888) (58-minute video)


#### Cheat Sheet

See the complete documentation.

| Command     | Description                                      |
|-------------|--------------------------------------------------|
| `tmux`      | Start the server                                  |
| `Ctrl+B D`  | Disconnect from the server                         |
| `tmux a`    | Reconnect to the server                           |
| `Ctrl+B C`  | Create a new window                              |
| `Ctrl+B N`  | Go to the next window                            |
| `Ctrl+B [`  | Enable copy mode for scrolling with mouse and page up/down keys |
| `Esc`       | Disable copy mode                                 |


#### Using tmux within a tmux-submitted task

If you use `tmux` to submit a task and attempt to launch `tmux` inside that task, you will get the error message "lost server". This is because the environment variable `$TMUX`, which points to the `tmux` server on the connection node, is propagated to the task. The value of the variable is therefore not valid. You can reset it with:

```bash
[name@server ~]$ unset TMUX
```

However, using two (or more) levels of `tmux` is not recommended. To send commands to a nested `tmux`, you must press the `Ctrl+B` keys twice; for example, to create a window, you must type `Ctrl+B Ctrl+B C`. Consider instead using GNU Screen (below) inside your tasks (if you use `tmux` on a connection node).


### GNU Screen

The `GNU Screen` program is another often-used terminal multiplexer. Create a detached terminal session with:

```bash
[name@server ~]$ screen -S <session name>
```

Give your sessions easy-to-remember names. To see the list of detached sessions on a node, use the command `screen -list`:

```bash
[name@server ~]$ screen -list
There is a screen on:
164133.foo (Attached) 1 Socket in /tmp/S-stubbsda.
```

To attach to one of your sessions, use:

```bash
screen -d -r <session name>
```
