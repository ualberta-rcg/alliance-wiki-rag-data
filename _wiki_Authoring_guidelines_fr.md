# Authoring Guidelines

This page is a translated version of the page [Authoring guidelines](https://docs.alliancecan.ca/mediawiki/index.php?title=Authoring_guidelines&oldid=174895) and the translation is 92% complete. Outdated translations are marked like this.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Authoring_guidelines&oldid=174895), français

## Who can contribute to this wiki?

If you have an Alliance account, you can contribute.  Our team's main task is to provide complete and accurate documentation, but we are in the Wikipedia era. If you notice an obvious problem like a broken link or a typo, you can of course correct it. If you wish, you can also write an article related to software you are familiar with. Our documentation team reviews submitted articles to verify their compliance with these guidelines.

Collaboration on the wiki is not anonymous. You must log in using your account credentials; this allows us to know who wrote or modified the content.

## Wiki Content

This wiki is not the place to display information that falls under the responsibility of the communications and marketing department, including any information for communication to the general public, the media, and funding agencies. Also, information concerning training and outreach activities is not suitable for the content of the technical documentation. Before publishing a page or modifying the content of the wiki, ask yourself the following questions:

* Is this information about the availability of a cluster or service? If so, has this cluster or service been announced? Otherwise, contact the communications and marketing department before publishing.
* Is it a status that changes daily (available, offline, under maintenance, etc.)? This information should appear on [https://status.alliancecan.ca/](https://status.alliancecan.ca/).
* Is the information intended for users or our technical teams? If it is intended for a technical team, it should be located on [https://wiki.computecanada.ca/staff/](https://wiki.computecanada.ca/staff/) rather than on [https://docs.alliancecan.ca/](https://docs.alliancecan.ca/).
* Does this information affect the security of our systems or the security of the data on our systems? If so, contact the information security department before publishing.
* Is the information intended for potential users rather than account holders? There is a gray area here: just like a potential user, an account holder might want to know the technical details related to our services and sites. However, if the information is only of interest to potential users, it should be found on [https://www.alliancecan.ca](https://www.alliancecan.ca) rather than on [https://docs.alliancecan.ca/](https://docs.alliancecan.ca/).
* It is appropriate to publish external links; see for example [Getting an account](example_link_needed).
* Does the information explain how to use an existing cluster, application, or service? If so, go ahead.


If you still have doubts:

* If you are employed by the Alliance, use the #rsnt-documentation channel in Slack;
* If you are not employed by the Alliance, contact [technical support](example_link_needed).

## Style Guide

As much as possible, avoid uploading PDF files. Instead, copy the selected text from a PDF and then modify it according to the wiki standards, including, for example, internal links to other pages or sections.

### Drafts

If you are developing a new page and it is not complete, you should mark the page as a draft by inserting `{{Draft}}`

### Writing

The style guide helps writers produce technical documentation that facilitates learning. Well-prepared documentation is pleasant for the reader and projects a positive image of the author.

There are several style guides for technical documentation. The Office québécois de la langue française lists a few.

No style guide exists for our wiki, but it is important to remember certain common practices:

* State one main idea per paragraph.
* Place information in order of importance.
* Address the reader directly. For example, "Click the button" rather than "The user must click the button".
* Use common and simple vocabulary as much as possible.
* Construct your sentences with verbs in the present tense.
* Use the active voice. For example, "The file contains the valid parameters" rather than "The valid parameters are contained in the file".
* Use the positive form. For example, "Answer YES" rather than "Do not answer NO".
* Use the right word.  Of course, synonyms make the text less boring, but they can create confusion for a new user or a user whose mother tongue is different from that of the text (for example, *machine*, *host*, *node*, *server*). The term *system* is often used generically: it can refer to a computer, a cluster, or even an environment or software. Be sure to use the right word to avoid any confusion.

#### Other Resources

* [Technical Writing courses from Google](example_link_needed)
* [Documentation guide from Write the Docs](example_link_needed)

### Layout

When in doubt, imitate the masters. Use the style of an existing page. If you can't find one on `docs.alliancecan.ca/`, look on Wikipedia.

As much as possible, keep images separate from text content. Do not use line breaks to adjust vertical spacing. Do not use tabs or spaces to indent a paragraph; do not add spaces at the end of a sentence. If this type of formatting is desirable, we will prepare stylesheets or templates, as needed.

Screenshots are useful, especially in the case of user guides or tutorials. However, full-screen captures included in the text impair readability. Place floating images to the right of the text. Also reduce the image size. If visibility is not acceptable, perhaps you should crop the image? You can also add the mention "Click to enlarge".

Use as few synonyms as possible. Of course, synonyms make the text less boring, but they can create confusion for a new user or a user whose mother tongue is different from that of the text (for example, *machine*, *host*, *node*, *server*).

Leave a blank line at the end of a section, before the title of the next section. The translation function uses the blank line and the title to delimit the translation units.

### Templates

Several [templates](example_link_needed) are available. Please use them as needed. We draw your attention in particular to the templates for [Including a command in a wiki page](example_link_needed) and [Including a source code file in a wiki page](example_link_needed).

## Translation

The page in the source language must be marked for translation. Anyone can translate a page marked for translation using the tools of the wiki extension *Translate*. You will find a tutorial [here](example_link_needed). A translated page can then be reviewed.

When a page is marked for translation, the Translate extension analyzes its content and divides it into translation units, which are, for example, a title, a paragraph, an image, or other. Discrete units are translated individually: thus, a modification to one unit has no effect on the rest of the page and it is possible to know the percentage of the page already translated or to be updated.

### Marking a page for translation

When you have finished writing a page, you must indicate that it is ready to be translated by following these steps:

* The content to be translated must be enclosed between the tags `<translate> </translate>`.
* Also use the tags `<translate> </translate>` to delimit the code that should not be translated.
* The tags `<translate> </translate>` are also used to isolate the wiki markup code (e.g., tables and tags).
* The tag `<languages />` must appear at the very beginning of the page. This will display a box at the top of the page containing the list of languages available for the page.
* In *View* mode, click on "Mark this page for translation".
* Review the translation units. Make sure the text is complete and that the programming code and wiki code (tables, tags, etc.) are excluded from the translation units.
* Select the priority language for translation; this is the target language.
* Click on "Mark this page for translation".

### Identifying modifications in a page marked for translation

It is recommended to mark a page for translation once the content in the source language is stable.

If a page that has already been translated does not contain any changes, avoid modifying codes such as `<!--T:3-->`, which are automatically generated codes. You should never edit or copy these codes.

Once the page has been corrected, mark the changes to be translated as follows:

* The new content to be translated must be enclosed between the codes `<translate> </translate>`.
* Also use the tags `<translate> </translate>` to delimit the code that should not be translated.
* The tags `<translate> </translate>` are also used to isolate the wiki markup code (e.g., tables and tags).
* In *View* mode, a message at the top of the page informs you that the page contains modifications made after it was marked for translation.
* Review the translation units. Make sure the text is complete and that the programming code and wiki code (tables, tags, etc.) are excluded from the translation units.
* Verify that the priority language is selected; this is the target language.
* Click on "Mark this page for translation".

If the modification you make to a translation unit in the source page has no impact on the target version, for example if you only correct a typo, check the box "Do not invalidate translations" and the target version will not be identified as needing to be updated.

### Translating code blocks

Content that is in the form of a programming language is not translated into another language. It is recommended to isolate code blocks with `</translate>` to mark the end of the text to be translated and the beginning of the code and then `<translate>` to mark the end of the code and the resumption of the text to be translated.

An excellent practice in programming is to add explanatory comments to the code itself. However, this information loses its value if it is not translated. There is no single solution that would work in all cases, but we offer the following suggestions:

* Place important comments outside the code blocks.
* Insert an index comment (e.g., NOTE 1, NOTE 2) to link the text to the corresponding line of code.
* If you are fluent in the other language and know the wiki's translation functions, you can translate the comments.

Always remember to insert comments in the code, but ask yourself if this information is important enough to be translated.

### Translating the sidebar

To add an item that is to be translated in the sidebar, use the following steps:

1. Add the new content to `MediaWiki:Sidebar`. Any item which should be translated should be added as either `some-tag` or, if it is a link, `{{int:some-tag}}`.
2. Add the tags to `MediaWiki:Sidebar-messages`.
3. Define the content of the tag in English on `MediaWiki:some-tag` (replace `some-tag` by the actual tag).
4. Translate the content of the tag on [this page](example_link_needed).

## List of available software

The tables on the wiki page [Available software](example_link_needed) are generated from module files in CVMFS. To add a link to a new page in the *Documentation* column, make a new entry in [https://github.com/ComputeCanada/wiki_module_bot/blob/main/module_wiki_page.json](https://github.com/ComputeCanada/wiki_module_bot/blob/main/module_wiki_page.json). Then add this modification to the final copy of the file.

Modifications may take up to six hours to appear on the wiki page.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Authoring_guidelines/fr&oldid=174896](https://docs.alliancecan.ca/mediawiki/index.php?title=Authoring_guidelines/fr&oldid=174896)"
