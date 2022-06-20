# Predicting Logerror
by Deangelo Bowen 
20 Jun 2022

---
# Overview

## Project Description:
- The focus of this project is to use the 2017 properties and predictions data for single unit / single family homes to uncover what the drivers of the error in the zestimate is.

## Project Goals:
- Construct a ML regression model with clustering techniques that can predict logerror for Single Unit/Family properties that had a transaction in 2017 using defining features for those properties.
- Find the Key Drivers of that logerror
- Make recommendations on what works or what doesn't work when predicting logerror.

---

## Executive Summary:
### When observing general home features such as bedrooms and bathrooms:

- There is some linear correlation between number of bedrooms and logerror.
- There is some linear correlation between number of bathrooms and logerror.

### When controlling for county:

- There is a significant difference between Los Angeles Logerror and the population logerror.
- There is a significant difference between Ventura Logerror and population logerror.
- There is no significant difference, or no difference between Orange County Logerror and the population logerror.

### When observing age of property:

- There is a linear correlation between the age of the property and logerror.

### When controlling for property type:

- Townhomes and Mobile home data sample were too small to perform any meaningful statistical operations.
- No linear correlation between type single family residential's year built, and logerror.
- Strong postive linear correlation between the year condiminums were built and logerror. (P value .96)

##### I would recommend pursuing further identifications of key drivers for logerror to potentially construct better accurate predictors as none of the current models beat the baseline.
  
---

## Data Dictionary
|Column | Description | Dtype|
|--------- | --------- | ----------- |
|bathrooms| number of bathrooms| float64|
|bedrooms| number of bedrooms| int64|
|property_age| calculate age of property| int64|
|home_amount| amount of taxes| float64|
|county| county location| String|
|sqare_feet| home sqare feet| int64|
|year_built| home year was built| int64|
|logerror| calculated margin of error| int64|

    
---
   
### Planning
   - Define goals
   - Determine audience and delivery format
   - Ask questions/formulate hypothesis
   - Determine the MVP

### Acquisition
   - Create a function that establishes connection to zillow_data in CodeUp mySQL
   - Create a function that holds your SQL query and reads results
   - Creating functing for caching data and stores as .csv for ease
   - Create and save in acquire.py so functions can be imported

   ### Preparation
   - Identifiy each of the features data types and manipulated relevant columns to appropriate dtypes.
   - Remove all irrelevant or duplicated columns.
   - Renamed columns to more appropriate and identifiable naming conventions.
   - Repeated these steps on the split data for future modeling.
    
   ### Exploration
   - Use the initial questions to guide the exploration process
   - Create visualizations to help identify drivers
   - Use statistical testing to confirm or deny hypothesis
   - Document answers to questions as takewaways
   - Utilize explore.py as needed for clean final report

 ### Model
   - Train model
   - Make predictions
   - Evaluate model
   - Compute accuracy
   - Utilize wrangle.py and explore.py as needed for clean final report
--- 

### Key Findings and Takeaway's Summary : 
### When observing general home features such as bedrooms and bathrooms:

- There is some linear correlation between number of bedrooms and logerror.
- There is some linear correlation between number of bathrooms and logerror.

### When controlling for county:

- There is a significant difference between Los Angeles Logerror and the population logerror.
- There is a significant difference between Ventura Logerror and population logerror.
- There is no significant difference, or no difference between Orange County Logerror and the population logerror.

### When observing age of property:

- There is a linear correlation between the age of the property and logerror.

### When controlling for property type:

- Townhomes and Mobile home data sample were too small to perform any meaningful statistical operations.
- No linear correlation between type single family residential's year built, and logerror.
- Strong postive linear correlation between the year condiminums were built and logerror. (P value .96)
---
## Results of Obeserving the Primary Goals:

- [x] #### Constructed a ML Regression model that can predict logerror.  
    - My the Lasso + Lars model had the best performace 
- [x] #### Find Key drivers of property tax value for single family homes.
    - Number of bedrooms and logerror.
    - Number of bathrooms and logerror.
    - The age of a property
    - Square Footage
    - Property Type Year Built
 
---
### Recommendations : 

- I would recommend to continue improving upon the baseline model as it works well enough given the current situation.
- I would recommend pursuing further identifications of key drivers for logerror to potentially construct better accurate predictors.
---
### Next Steps:
- Conduct more feature engineering and diversifying my clustering
- Create more robust models that continue to race towards beating the baseline model.
---

### To Recreate this Project:
   - You will need an env file with database credentials saved to your working directory database credentials with CodeUp database (username, password, hostname) 
   - Create a gitignore with env file inside to prevent sharing of credentials
   - Download the acquire.py and prepare.py (model.py and explore.py are optional) files to working directory
   - Create a final notebook to your working directory
   - Review this README.md
   - Libraries used are pandas, matplotlib, Scipy, sklearn, seaborn, and numpy
   - Run final_notebook.ipynb
