This is the content for the talk "Going from Notebooks to Scalable Systems", presented at [PyCon US 2025](https://us.pycon.org/2025/).

The data used in this talk is the ["Palmer Penguins" dataset](https://allisonhorst.github.io/palmerpenguins/).

## About me

I'm the author of "Software Engineering for Data Scientists", published by O'Reilly Media in 2024. It's a guide for data professionals to level up their Python coding skills, especially for early- to mid-career folks. You can read much more about the topics in this talk in my book!

* Read it on [the O'Reilly Platform](https://www.oreilly.com/library/view/software-engineering-for/9781098136192/)
* Buy it on [Amazon](https://www.amazon.com/dp/1098136209)
* Buy it [from your local bookstore](https://bookshop.org/p/books/software-engineering-for-data-scientists-from-notebooks-to-scalable-systems-catherine-nelson/21142977)


### Social media

* Connect on [LinkedIn](https://www.linkedin.com/in/catherinenelson1/)
* Follow me on [BlueSky](https://bsky.app/profile/catnelson.bsky.social)

## Files in this repository

*from_notebooks.pdf* contains the slides for the PyCon 2025 talk.

*penguins_notebook.ipynb* is a typical "data science" style notebook, with few functions and many instances where the data is displayed. It contains a training pipeline for a model to predict penguin species.

*penguins_refactored.py* is the same code, but refactored into a script that is robust and reproducible.

*test_penguins_refactored.py* contains unit tests for the functions in _penguins_refactored.py_

_requirements.txt_ contains the dependencies for this code.

The folder *files_for_presentation* contains the draft files I used to prepare the slides.


## Links and references

### Tools in this talk

- [Jupytext](https://jupytext.readthedocs.io/en/latest/) can convert notebooks to paired Python scripts.

- Use [nbconvert](https://nbconvert.readthedocs.io/en/latest/usage.html#convert-script) to convert a notebook to a script.

### Further reading

Another great perspective on this topic: https://transferlab.ai/trainings/beyond-jupyter/ 

How to make great slides: https://ines.io/blog/beginners-guide-beautiful-slides-talks/ 

