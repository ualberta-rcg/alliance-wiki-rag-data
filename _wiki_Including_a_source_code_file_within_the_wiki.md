# Including Source Code within the Wiki

As explained on the page [Including source code within the wiki](link_to_page_if_available), you can include source code within the wiki using the `<syntaxhighlight></syntaxhighlight>` tag.  If you want to separate the code a bit more from the rest of the text, you can use the `{{File}}` template. This template takes as arguments the file name (`name` parameter), the language of the file (`lang` parameter), and the content of the file (`contents` parameter). The default language for this template is `bash`.

For example:

```
{{File
  |name=myfile.sh
  |lang="bash"
  |contents=
#!/bin/bash
echo "this is a bash script"
}}
```

results in:

File: myfile.sh
```bash
#!/bin/bash
echo "this is a bash script"
```

## Special Characters: Pipe, Equals

Certain characters that frequently appear in bash scripts are also meaningful to the MediaWiki template parser.

If the source code contains a pipe character (`|`), replace it with `{{!}}`.

In some circumstances, you may find it necessary to replace the equal sign (`=`), with `{{=}}`.


## Displaying Line Numbers

To display line numbers, you can add the option `|lines=yes`. For example:

```
{{File
  |name=monfichier.sh
  |lang="bash"
  |lines=yes
  |contents=
#!/bin/bash
echo "this is a bash script"
}}
```

results in:

File: myfile.sh
```bash
#!/bin/bash
echo "this is a bash script"
```

**(Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Including_a_source_code_file_within_the_wiki&oldid=46635](https://docs.alliancecan.ca/mediawiki/index.php?title=Including_a_source_code_file_within_the_wiki&oldid=46635)")
