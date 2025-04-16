# Heritage Housing Issues

## Introduction
Lydia Doe has recently inherited four properties located in Ames, Iowa, USA, from her late great-grandfather. While she is well-versed in the real estate market of her home country, Belgium, she is concerned that her understanding may not apply to the housing market in Iowa. Factors that increase a property's value in Belgium may not hold the same influence in Ames.

In order to make informed decisions and ensure the best possible return on the properties, Lydia seeks help from a Data Practitioner. Her primary goal is to accurately estimate the value of these homes and avoid financial loss due to mispricing.


## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

## Business Requirements

Lydia's objectives for this data-driven project are twofold:

1. She wants to understand which housing attributes most strongly influence sale prices in Ames. To achieve this, she expects clear visualizations that show the correlation between house features and their final selling price.

2. She needs to predict the expected sale price for her four inherited houses based on their characteristics. Additionally, she wishes to use this tool to estimate the price of any other property in Ames she might consider buying in the future.

## Hypothesis and how to validate?

### Hypothesis 1:
Homes with larger overall living areas (`GrLivArea`), higher material quality (`OverallQual`), and newer construction years tend to have higher sale prices.

### Hypothesis 2:
Features such as garage size, kitchen quality, and the presence of finished basement areas significantly contribute to the value of a property.

---

### Validation Strategy:

- Perform **correlation analysis** and create **data visualizations** (e.g., scatterplots and heatmaps) to explore the relationships between individual features and `SalePrice`.

- Use **regression modeling** to quantify the impact of specific variables on the target variable (`SalePrice`) and evaluate the model’s accuracy using metrics such as **R² score** and **RMSE** (Root Mean Squared Error).

- Test the trained model by running predictions on **Lydia’s four inherited properties** to verify real-world performance and usability.


## The rationale to map the business requirements to the Data Visualisations and ML tasks

Business Requirement 1:
The client wants to understand how different property features relate to the final sale price.

- Data Visualisation Task:
To meet this requirement, various data visualisations will be used, such as:

  - Correlation heatmaps to identify the most strongly related numerical features.

  - Scatter plots to show how continuous variables (e.g., GrLivArea, GarageArea) influence sale price.

  - Boxplots and bar charts for categorical variables (e.g., KitchenQual, OverallQual).

These visualisations will help reveal patterns, trends, and key influencing factors that determine house value in Ames.

Business Requirement 2:
The client wants to predict the sale price of her four inherited houses and any other similar property.

- Machine Learning Task:
A regression model will be built to predict house sale prices based on selected features. The model will be trained on historical housing data and tested for accuracy using standard regression metrics.

- This model will be integrated into a Streamlit Web App, enabling the user to input house attributes and receive a predicted sale price in real-time.

## ML Business Case

To support the second business requirement, a supervised machine learning regression model will be developed with the following framework:

- Goal:
Predict the sale price of a house in Ames, Iowa based on its attributes (such as living area, build year, quality ratings, etc.).

- Model Type:
A regression model will be used. Several algorithms will be tested (e.g., Linear Regression, Random Forest, Gradient Boosting), and the best-performing model will be selected based on evaluation metrics.

- Inputs (Features):
Selected numerical and categorical variables that show strong correlation with SalePrice. Examples include:

  - GrLivArea, OverallQual, YearBuilt, GarageArea, KitchenQual, etc.

- Output:
The predicted SalePrice of a given house.

Evaluation Metrics:

- R² Score: Measures how well the model explains variability in the target variable.

- RMSE (Root Mean Squared Error): Indicates average prediction error.

- Comparison of actual vs. predicted prices using visual plots.

Business Impact:
The ML model will help Lydia make informed pricing decisions on her inherited properties and assist in evaluating future investment opportunities. Accurate predictions will reduce financial risk and support data-driven real estate decisions.

## Dashboard Design

### Page 1: Quick Project Summary

This introductory page provides a comprehensive overview of the project and its key components. It includes:

- A brief summary of the project’s objective and scope  
- A glossary explaining key project terms and technical jargon  
- A description of the dataset used, including its origin and structure  
- A clear statement of the two business requirements that guide the project’s direction  

---

### Page 2: Sales Price Correlation Study

This page addresses **Business Requirement 1**: to explore how various house attributes relate to the final sale price.

- A button labeled **"Inspect House Price Dataset"** allows the client to view the raw dataset, including dimensions and sample entries  
- A summary of insights regarding the top 10 most correlated features with sale price is displayed  
- Two additional buttons enable the user to toggle between:
  - A heatmap showing overall feature correlations  
  - Scatter plots of key variables against SalePrice for a more detailed examination  

---

### Page 3: House Sales Price Predictor

This page is designed to fulfill **Business Requirement 2**: to predict the sale prices of Lydia’s four inherited properties and any additional property.

- The attributes of each of the four houses can be entered, and their respective predicted sale prices are displayed  
- The total estimated sale price of the four properties is also shown to provide Lydia with a clear financial overview  
- The **"Predict Sales Price"** button feeds the input data into the machine learning pipeline, triggering a real-time prediction output  

---

### Page 4: Project Hypotheses

This page summarizes and validates the project hypotheses developed during the exploratory data analysis.

- **Hypothesis 1: Property size has a significant impact on sale price.**  
  Analysis of area-related variables suggests that larger properties are generally valued higher.

- **Hypothesis 2: Newer houses tend to command higher prices.**  
  A historical trend analysis shows that recently built homes are often priced above older ones.

- **Hypothesis 3: Kitchen quality is a key factor in pricing.**  
  A strong correlation between kitchen quality and sale price indicates that well-equipped kitchens may increase a property’s market value.

---

### Page 5: ML – Predict Sales Price

This page presents the machine learning pipeline and summarizes its key components and results.

- A breakdown of the steps involved in building and training the ML pipeline  
- A plot or table displaying the **feature importance**, helping users understand which variables most influenced predictions  
- Key **performance metrics** of the model (e.g., R², RMSE) to evaluate its reliability and accuracy  
- Final reflections and insights based on the model's training and prediction outcomes  


## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment

### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the .python-version Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)


* In case you would like to thank the people that provided support through this project.

