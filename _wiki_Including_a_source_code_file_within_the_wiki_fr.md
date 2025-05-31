# Including a Source Code File Within the Wiki

As mentioned on the page [Including source code in the wiki](link-to-english-page-needed), the `<syntaxhighlight></syntaxhighlight>` tags are used to include code.

If you want the code to be separate from the text, use the template `{{File}}`. This template takes the name (parameter `name`), the language (parameter `lang`), and the content (parameter `contents`) of the file as arguments. This template uses bash as the default language.

For example,

```
{{Fichier
  |name=myfile.sh
  |lang="bash"
  |contents=
#!/bin/bash
echo "ceci est un script bash"
}}
```

gives the following result:

File: myfile.sh
```bash
#!/bin/bash
echo "ceci est un script bash"
```

## Special Characters: Vertical Bar and Equal Sign

Bash scripts often contain characters that also have meaning for the MediaWiki parser.

If the source code contains a vertical bar (the character `|`), replace it with `{{!}}`.

In some cases, you need to replace the equal sign (the character `=`) with `{{!}}`.

## Displaying Line Numbers

To display line numbers, add the option `lines=yes`, for example:

```
{{Fichier
  |name=monfichier.sh
  |lang="bash"
  |lines=yes
  |contents=
#!/bin/bash
echo "ceci est un script bash"
}}
```

gives the following result:

File: myfile.sh
```bash
#!/bin/bash
echo "ceci est un script bash"
```
