# Perl

Other languages: English, français

## Description

Perl is a free, interpreted programming language with a vast library of contributed packages accumulated over its 25+ years of existence.  Its strengths include string manipulation, database access, and portability (according to [this article](link_to_article_needed)). Its weaknesses are its poor performance and the ease with which one can write obscure and illegible code. By design, Perl offers several different ways to accomplish the same task. Many programmers have adopted this language and write code that is very compact but difficult to decipher.


## Loading the Interpreter

The Perl language is available on Compute Canada's servers via a module. Load it like any other module:

```bash
[name@server ~]$ module spider perl
```

This shows installed versions. Then load a specific version:

```bash
[name@server ~]$ module load perl/5.30.2
```


## Installing Packages

Many Perl packages are installable via the Comprehensive Perl Archive Network (CPAN) using the `cpan` tool.  This requires correct initialization to install packages in your home directory.  Many Perl packages are developed using the GCC compiler family; load a `gcc` module beforehand:

```bash
[name@server ~]$ module load gcc/9.3.0
```

### Initial Configuration for Package Installation

The first time you run `cpan`, it will ask if you want to configure settings automatically. Respond `yes`.

```bash
[name@server ~]$ cpan

...

Would you like me to configure as much as possible automatically? [yes]
...
What approach do you want? (Choose 'local::lib', 'sudo' or 'manual') [local::lib]
...
```

The `cpan` utility will offer to append environment variable settings to your `.bashrc` file; agree to this.  Type `quit` to exit `cpan`. Restart your shell for the new settings to take effect before installing modules.


### Package Installation

After initial configuration, install packages from CPAN (25,000+ available). For example:

```bash
[name@server ~]$ cpan

Terminal does not support AddHistory.

cpan shell -- CPAN exploration and modules installation (v2.11)
Enter 'h' for help.

cpan[1]> install Chess

...
Running install for module 'Chess'
Fetching with LWP: http://www.cpan.org/authors/id/B/BJ/BJR/Chess-0.6.2.tar.gz
Fetching with LWP: http://www.cpan.org/authors/id/B/BJ/BJR/CHECKSUMS
Checksum for /home/stubbsda/.cpan/sources/authors/id/B/BJ/BJR/Chess-0.6.2.tar.gz ok
Scanning cache /home/stubbsda/.cpan/build for sizes ............................................................................DONE
'YAML' not installed, will not store persistent state
Configuring B/BJ/BJR/Chess-0.6.2.tar.gz with Makefile.PL
Checking if your kit is complete...
Looks good
...
Running make for B/BJ/BJR/Chess-0.6.2.tar.gz
...
Running make test PERL_DL_NONLAZY = 1 "/cvmfs/soft.computecanada.ca/nix/store/g8ds64pbnavscf7n754pjlx5cp1mkkv1-perl-5.22.2/bin/perl" "-MExtUtils::Command::MM" "-MTest::Harness" "-e" "undef *Test::Harness::Switches; test_harness(0, 'blib/lib', 'blib/arch')" t/*.t
t/bishop.t ......... ok
t/board.t .......... ok
t/checkmate.t ...... ok
t/game.t ........... ok
t/king.t ........... ok
t/knight.t ......... ok
t/movelist.t ....... ok
t/movelistentry.t .. ok
t/pawn.t ........... ok
t/piece.t .......... ok
t/queen.t .......... ok
t/rook.t ........... ok
t/stalemate.t ...... ok
All tests successful.
Files = 13 , Tests = 311 , 3 wallclock secs ( 0.14 usr  0.05 sys +  2.49 cusr  0.20 csys =  2.88 CPU)
Result: PASS
...
Installing /home/stubbsda/perl5/man/man3/Chess::Piece::Knight.3
Installing /home/stubbsda/perl5/man/man3/Chess.3
Installing /home/stubbsda/perl5/man/man3/Chess::Piece::Bishop.3
Installing /home/stubbsda/perl5/man/man3/Chess::Board.3
Appending installation info to /home/stubbsda/perl5/lib/perl5/x86_64-linux-thread-multi/perllocal.pod
BJR/Chess-0.6.2.tar.gz /cvmfs/soft.computecanada.ca/nix/var/nix/profiles/16.09/bin/make install -- OK

cpan[2]>
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Perl&oldid=97027](https://docs.alliancecan.ca/mediawiki/index.php?title=Perl&oldid=97027)"
