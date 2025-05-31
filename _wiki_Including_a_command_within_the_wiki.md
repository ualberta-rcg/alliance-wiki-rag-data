# Including a Command Within the Wiki

Other languages: English, français

To include a command within the wiki, use the `{{Command}}` template. This template detects bash syntax. For example, the code:

```
{{Command|cd src; make && make install; cd ..}}
```

results in:

```
[name@server ~]
$ cd src; make && make install; cd ..
```

## Special Characters "=" and "|"

Since `{{Command}}` is a template, the "=" and "|" signs are interpreted by the wiki.  To include an equality sign, use the meta-template `{{=}}`. For example, the code:

```
{{Command|./configure --prefix{{=}}$HOME && make && make install}}
```

results in:

```
[name@server ~]
$ ./configure --prefix=$HOME && make && make install
```

To include a pipe symbol, use `{{!}}`.


## Including a Set of Commands

Use the `{{Commands}}` template to include a set of commands. Write each command on a separate line, prepending the "|" character. For example, the code:

```
{{Commands
|cd src
|make
|make install
|cd ..
}}
```

results in:

```
[name@server ~]
$ cd src
[name@server ~]
$ make
[name@server ~]
$ make install
[name@server ~]
$ cd ..
```

## Modifying the Command Prompt

Modify the command prompt by including a `prompt` argument to the template. For example,

```
{{Command|prompt=[name@briaree ~]|cd src; make && make install; cd ..}}
```

results in:

```
[name@briaree ~]
cd src; make && make install; cd ..
```

Similarly,

```
{{Commands
|prompt=[name@briaree $]
|cd src
|make
|make install
|cd ..
}}
```

results in:

```
[name@briaree $]
cd src
[name@briaree $]
make
[name@briaree $]
make install
[name@briaree $]
cd ..
```

## Displaying the Result of a Command

Display the result of a command (only one) by adding the `result` option. For example,

```
{{Command
|df -h .
|result=
Sys. de fich.         Tail. Occ. Disp. %Occ. Monté sur
/lustre2/home         516T  340T  150T  70% /home
}}
```

results in:

```
[name@server ~]
$ df -h .
Sys. de fich.         Tail. Occ. Disp. %Occ. Monté sur
/lustre2/home         516T  340T  150T  70% /home
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Including_a_command_within_the_wiki&oldid=67931](https://docs.alliancecan.ca/mediawiki/index.php?title=Including_a_command_within_the_wiki&oldid=67931)"
