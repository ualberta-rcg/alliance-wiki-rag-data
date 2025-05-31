# Prolonging Terminal Sessions

Other languages: English, français

To work on our clusters, most users will need to use SSH to connect to the cluster for job submission and monitoring, editing files, and so forth. Keeping this SSH connection alive for a long period of time, ranging from hours to days, may be necessary for some users. This page proposes a variety of techniques for keeping such a terminal session alive.

## SSH Configuration

One simple solution is to modify the configuration of your SSH client to prolong the connection.  On MacOS and Linux, the client configuration is found in `$HOME/.ssh/config`, while in Windows it is located in `C:\Users\<username>\.ssh\config`. Note that the file may not exist initially, so you will need to create it. You should add the lines:

```
Host *
    ServerAliveInterval 240
```

This addition will ensure the transmission of a sign-of-life signal over the SSH connection to any remote server (such as an Alliance cluster) every 240 seconds (four minutes), which should help to keep your SSH connection alive even if it is idle for several hours.

## Terminal Multiplexers

The programs `tmux` and `screen` are examples of a terminal multiplexer—a program which allows you to detach your terminal session entirely, where it will keep on running on its own until you choose to reattach to it. With such a program, you can log out from the cluster, turn off the workstation, or hibernate the laptop you use to connect to the cluster. When you're ready to start working again the next day, reattach to your session and start from right where you left off.

### Login Node Dependency

Each of our clusters has several login nodes, and your `tmux` or `screen` session is specific to a login node. If you wish to reattach to a session, you must ensure you're connected to the right login node, which of course means remembering which login node you were using when you started `tmux` or `screen`. Login nodes may also occasionally be rebooted, which will kill any detached terminal sessions on that node.

### tmux

The `tmux` software is a terminal multiplexer, allowing multiple virtual sessions in a single terminal session. You can thus disconnect from an SSH session without interrupting its process(es).

Here are some introductions to tmux:

*   "The Tao of tmux", an online book
*   "Getting Started With TMUX", a 24-minute video
*   "Turbo boost your interactive experience on the cluster with tmux", a 58-minute video

#### Cheat Sheet

For a complete reference, see [this page](link_to_tmux_reference_page).

| Command     | Description                                      |
|-------------|--------------------------------------------------|
| `tmux`      | Start a server                                   |
| `Ctrl+B D`  | Disconnect from server                           |
| `tmux a`    | Reconnect to server                             |
| `Ctrl+B C`  | Create a new window                             |
| `Ctrl+B N`  | Go to next window                               |
| `Ctrl+B [`  | Enable "copy" mode, allowing to scroll with mouse and Page-Up/Page-Down |
| `Esc`       | Disable "copy" mode                             |


#### Launch tmux Inside a Job Submitted Through tmux

If you submit a job with `tmux` and try to start `tmux` within the same job, you will get the "lost server" error message. This happens because the `$TMUX` environment variable pointing to the tmux server is propagated to the job. The value of the variable is not valid, and you can reset it with:

```bash
[name@server ~]$ unset TMUX
```

However, nested use of `tmux` is not recommended. To send commands to a nested `tmux`, one has to hit `Ctrl+B` twice; for example, to create a new window, one has to use `Ctrl+B Ctrl+B C`. Consider using `GNU Screen` inside your job (if you are using `tmux` on a login node).

### GNU Screen

The `GNU Screen` program is another widely used terminal multiplexer. To create a detached terminal session, you can use the following command:

```bash
[name@server ~]$ screen -S <session name>
```

It's a good idea to give a descriptive name to your terminal sessions, making it easier to identify them later. You can use the command `screen -list` to see a list of your detached terminal sessions on this node:

```bash
[name@server ~]$ screen -list
There is a screen on:
164133.foo (Attached) 1 Socket in /tmp/S-stubbsda.
```

You can attach to one of your sessions using the command `screen -d -r <session name>`.


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Prolonging_terminal_sessions&oldid=139813")**
