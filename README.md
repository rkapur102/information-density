# Project Template

This is an example of how a project repo should be organized.

It contains several subdirectories that will contain standard elements of almost every project:

- `analysis`: This subdirectory will typically contain Python or R notebooks for making visualizations and statistical analyses. `/analysis/plots/` is the path to use when saving out `.pdf`/`.png` plots, a small number of which may be then polished and formatted for figures in a publication.
- `data`:  This subdirectory is where you put the raw and processed data from behavioral experiments and computational simulations. These data serve as input to `analysis`. *Important: Before pushing any csv files containing human behavioral data to a public code repository, triple check that these data files are properly anonymized. This means no bare Worker ID's.*
- `experiments`: If this is a project that will involve collecting human behavioral data, this is where you want to put the code to run the experiment. If this is a project that will involve evaluation of a computational model on a task, this is also where you want to put the task code (which imports the `model`).
- `model`: If this is a cognitive modeling project, this is where you want to put code for running the model. If this project involves training neural networks, you would put training scripts in here.

# Project documentation 

## Project Log

When we spin up a new project, the first thing we'll do to collect our thoughts is to create a [Notion project](https://www.notion.so/social-interaction-lab/010f6821fc4e4aa1b7ec07716fd6cdc1?v=028218a3e35a4c079194b04b347a4d09&pvs=4) or a Google Doc to function as a running "log" of project updates and meeting notes. 
The point is to have a file format that is easy to share and flexible in format. 
This Google Doc / Notion page is also where you should take notes during our meetings, and collect high-level TODO items, especially those that are not immediately actionable. 

## Preregistration

Once we are in the later stages of desigining a new human behavioral experiment and preparing to run our first pilot, we will write up a pre-registration and either put it under version control within the project repo OR post it to the [Open Science Framework](https://osf.io/). We subscribe to the philosophy that ["pre-registrations are a plan, not a prison."](https://www.cos.io/blog/preregistration-plan-not-prison) They help us transparently document our thinking and decision-making process both for ourselves and for others, and will help us distinguish between confirmatory and exploratory findings. We do not believe that there is a single best way to write a pre-registration, but in many cases a more detailed plan will help us to clarify our experimental logic and set up our analyses accordingly (i.e., what each hypothesis predicts, which measures and analyses we will use to evaluate each relevant hypothesis). 

## Manuscripts 

When we are preparing to write up a manuscript (or a conference paper), we will start an [Overleaf](https://www.overleaf.com/) project. 
This is where you will want to place your LaTeX source `.tex` files for your paper and your publication-ready figures as high-resolution `.pdf` files in the `figures` directory. 
We typically format and fine-tune our figures using Adobe Illustrator.
