# Getting Started with the Compute Canada HPC Systems

## What do you want to do?

If you don't already have an account, see [Apply for an account](link-to-apply-for-account).  Information on [Multifactor authentication](link-to-multifactor-authentication) is also available.  Frequently asked questions are answered in the [Frequently Asked Questions about the CCDB](link-to-faq).


If you are an experienced HPC user and are ready to log onto a cluster, you probably want to know:

*   What [systems](#what-systems-are-available) are available?
*   What [software](link-to-software-info) is available, and how [environment modules](link-to-environment-modules) work?
*   How to [submit jobs](link-to-submit-jobs)?
*   How the [filesystems](link-to-filesystems) are organized.

If you are new to HPC, you can:

*   Read about how to connect to our HPC systems with [SSH](link-to-ssh-info).
*   Read an [introduction to Linux](link-to-linux-intro) systems.
*   Read about how to [transfer files](link-to-file-transfer) to and from our systems.


If you want to know which software and hardware are available for a specific discipline, a series of discipline guides is in preparation. At this time, you can consult the guides on:

*   [AI and Machine Learning](link-to-ai-ml-guide)
*   [Bioinformatics](link-to-bioinformatics-guide)
*   [Biomolecular simulation](link-to-biomolecular-simulation-guide)
*   [Computational chemistry](link-to-computational-chemistry-guide)
*   [Computational fluid dynamics (CFD)](link-to-cfd-guide)
*   [Geographic information systems (GIS)](link-to-gis-guide)
*   [Visualization](link-to-visualization-guide)

If you have hundreds of gigabytes of data to move across the network, read about the [Globus](link-to-globus) file transfer service.

Python users can learn how to [install modules in a virtual environment](link-to-python-virtualenv). R users can learn how to [install packages](link-to-r-packages).

If you want to experiment with software that doesn’t run well on our traditional HPC clusters, please read about [our cloud resources](link-to-cloud-resources).

For any other questions, you might try the Search box in the upper right corner of this page, the main page for [our technical documentation](link-to-technical-docs) or [contact us](link-to-contact-us) by email.


## What systems are available?

Six systems were deployed in 2016-2018: Arbutus, Béluga, Narval, Cedar, Graham, and Niagara.  In 2025 four of these are being replaced; see [Infrastructure renewal](link-to-infrastructure-renewal) for more on this.

*   **Arbutus:** A cloud site, which allows users to launch and customize virtual machines. See [Cloud](link-to-cloud-info) for how to obtain access to Arbutus.
*   **Béluga, Cedar, Narval, and Graham:** General-purpose clusters composed of a variety of nodes including large memory nodes and nodes with accelerators such as GPUs. You can log into any of these using SSH. A `/home` directory will be automatically created for you the first time you log in.
*   **Niagara:** A homogeneous cluster designed for large parallel jobs (>1000 cores). To obtain access to Niagara, visit the [Available Services](link-to-available-services) page.

Your password to log in to all new national systems is the same one you use to log into CCDB. Your username will be displayed at the top of the page once you've logged in.


## What training is available?

Most workshops are organized by the Alliance's regional partners; both online and in-person training opportunities exist on a wide variety of subjects and at different levels of sophistication. We invite you to consult the following regional training calendars and websites for more information:

*   **WestDRI (Western Canada Research Computing covering both BC and the Prairies regions):**
    *   [Training Materials website](link-to-westdri-training) - click on "Upcoming sessions" or browse the menu at the top for recorded webinars
*   **UAlberta ARC Bootcamp:** Videos of previous sessions available [link-to-ualberta-arc-bootcamp]
*   **SHARCNET:**
    *   [Training Events Calendar](link-to-sharcnet-calendar)
    *   [YouTube Channel](link-to-sharcnet-youtube)
*   **Online Workshops:** [link-to-online-workshops]
*   **SciNet:**
    *   [SciNet Education Site](link-to-scinet-education)
    *   [SciNet YouTube Channel](link-to-scinet-youtube)
*   **Calcul Québec:**
    *   [Workshops](link-to-calculquebec-workshops)
    *   [Training information](link-to-calculquebec-training)
*   **ACENET:**
    *   [Training information](link-to-acenet-training)
    *   [ACENET YouTube Channel](link-to-acenet-youtube)

One can also find a shared calendar of [upcoming workshops](link-to-upcoming-workshops).


## What system should I use?

This question is hard to answer because of the range of needs we serve and the wide variety of resources we have available. If the descriptions above are insufficient, contact our [technical support](link-to-technical-support).

In order to identify the best resource to use, we may ask specific questions, such as:

*   What software do you want to use?
*   Does the software require a commercial license?
*   Can the software be used non-interactively? That is, can it be controlled from a file prepared prior to its execution rather than through the graphical interface?
*   Can it run on the Linux operating system?
*   How much memory, time, computing power, accelerators, storage, network bandwidth, and so forth—are required by a typical job? Rough estimates are fine.
*   How frequently will you need to run this type of job?

You may know the answer to these questions or not. If you do not, our technical support team is there to help you find the answers. Then they will be able to direct you to the most appropriate resources for your needs.


**(Please replace bracketed `link-to-...` placeholders with actual links.)**
