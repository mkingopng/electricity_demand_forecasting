# Welcome to ZZSC9020 GitHub repository for group Team-O

This GitHub repository is the main point of access for students and lecturers of the ZZSC9020 capstone course. 

In this repository, you will find the data to start developing your project. Also, we will use the repository to share code, documentation, data, models and other resources between the group members and course lecturers.

Complete the information below regarding your group.

## Group and project information

### Group members and zIDs
- Noel Singh (zID1) - role tbc
- Reuben Bowell (z5382909) - role tbc
- William Stephan (z3404800) - role tbc
- Michael Kingston (z5372750) - role tbc

### Brief project description
to improve the accuracy of short-term electricity demand forecasting by:
- supplemental data containing explanatory variables in addition to temperature
- up-sampling the existing dataset
- define a naive baseline model, and establish historic forecast accuracy
- testing a range of model architectures, ranging from statistical methods 
  like ARIMA, to Bayesian approaches, to machine learning models like 
  Random Forest and GBDT.

## Repository structure
The repository has the following folder structure:
- **agendas**: agendas for each weekly meeting with lecturers (left 24h before the next meeting)
- **checklists**: teamwork checklist or a link to an account in a project task management tool
- **data**: datasets for analysis
- **gantt_chart**: Gantt chart or a link to an account in a project task management tool
- **minutes**: minutes for each meeting (left not more than 24h after the corresponding meeting)
- **report**: RMarkdown or Jupyter notebook report in progress
- **src**: source code


## Poetry
[Poetry](https://python-poetry.org/docs/) is a Python dependency manager. 
It takes advantage of PEP518, which introduces pyproject.toml as a new way to 
specify build requirements. The goal was to make the replicability of 
developers' environments easy to replicate, sync and maintain across team 
members.

## Basic Use:
1. install poetry on your computer, by following the instructions [here](https://github.com/python-poetry/install.python-poetry.org)

2. open a terminal and run:
  ```bash
  cd path/to/project/team-O
  poetry init
  ```
  poetry will then configure your environment based on the pyproject.toml in 
  the repository.

3. to install a new package run:
   ```bash
   poetry add <package-name>
   ```
   
4. to update your envirnnment to sync with repository, or to update a 
   dependency to the latest version run:

   ```bash
   poetry update
   ```

