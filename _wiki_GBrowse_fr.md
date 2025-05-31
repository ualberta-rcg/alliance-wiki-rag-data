# GBrowse

This page is a translated version of the page GBrowse and the translation is 100% complete.

Other languages: English, français

## Introduction

GBrowse is a tool for manipulating and visualizing genomic data through a web interface. It consists of a database combined with interactive web pages. GBrowse is available on Cedar and can be installed from [https://gateway.cedar.computecanada.ca](https://gateway.cedar.computecanada.ca).

Our installation differs from that described on the GBrowse webpage, particularly with regard to authorization and authentication.

## Accessing GBrowse

Our technical team will create a shared account for each research group requesting access to GBrowse. Data and configuration files will be readable by all members of the group. The principal investigator must contact [technical support](link-to-support-page-needed) to request the creation of the shared account and attest that they understand the terms of use of a shared account.

You must also indicate the name of the database used on Cedar. If you do not have a database account, see the Database Servers page (link-to-database-servers-page-needed).

## Installation

### Configuration Files

To make them visible to all members of the group, place the configuration files in the directory:

`/project/GROUPID/gbrowse/USERNAME/conf`

where `GROUPID` is the group identifier and `USERNAME` is your username. We will create a symbolic link from `${HOME}/gbrowse-config/` to this directory. Please ensure that group read permissions for these files are enabled.

### Database Connection

**With MySQL:** The GBrowse configuration files must contain:

```
[username_example_genome:database]
db_adaptor    =     Bio::DB::SeqFeature::Store
db_args       =    -adaptor DBI::mysql -dsn DATABASE;mysql_read_default_file=/home/SHARED/.my.cnf
```

where `DATABASE` is the database name and `SHARED` is the shared account. The `.my.cnf` text file is created by our technical team and contains the information needed to connect to MySQL.

**With Postgres:** The GBrowse configuration files must contain:

```
[username_example_genome:database]
db_adaptor    = Bio::DB::SeqFeature::Store
db_args       =  -adaptor DBI::Pg -dsn  =  dbi:Pg:dbname=DATABASE
```

where `DATABASE` is the database name.


## Usage

### Data Files

It is not necessary to upload `.bam` files to visualize them. For GBrowse to directly read `.bam` files:

*   The files must be copied to your `/project` directory and be readable by the group.
*   The directory containing the `.bam` files must have the `setgid` bit and group-execute permission enabled; i.e., the result of `ls –l` must have a lowercase `s` in place of the group execute `x`.
*   The group ownership of the `.bam` file must indicate the group name and not the username, for example `jsmith:def-kjones` rather than `jsmith:jsmith`.
*   The path to the `.bam` file must be modified in the configuration file, for example:

```
[example_bam:database]
db_adaptor        = Bio::DB::Sam
db_args           = -bam /project/GROUPID/USERNAME/gbrowse_bam_files/example_file.bam
search options    = default
```

### Uploading Files to the Database

With BioPerl, run:

```bash
module load bioperl/1.7.1
bp_seqfeature_load.pl -c –d DATABASE:mysql_read_default_file=/home/USERNAME/.my.cnf \
   example_genomic_sequence.fa header_file
```

where `DATABASE` is the database name, `example_genomic_sequence.fa` is the FASTA file containing the complete genome, and `header_file` contains details on the length of the chromosomes.

Here is an example of a header file:

```
##sequence-region I 1 15072434
##sequence-region II 1 15279421
##sequence-region III 1 13783801
##sequence-region IV 1 17493829
##sequence-region V 1 20924180
##sequence-region X 1 17718942
##sequence-region MtDNA 1 13794
```

Do not run this file on a head node, but execute these commands through the scheduler.

Once the data is uploaded to the database, you must grant read access to the shared account (`SHARED`) for GBrowse to be able to read the database; see Sharing your MySQL data (link-to-sharing-data-page-needed).


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=GBrowse/fr&oldid=123956](https://docs.alliancecan.ca/mediawiki/index.php?title=GBrowse/fr&oldid=123956)"
