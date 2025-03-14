# From Ingredients to Minutes: How Nutrition and Complexity Drive Cooking Time

Authors: Aarshia Gupta (aag022@ucsd.edu) and Nanda Payyappilly (npayyappilly@ucsd.edu)

---

## Overview

This data science project, conducted at UCSD, focuses on predicting whether a recipe has a long or short cooking time based on its **nutritional composition and preparation complexity**. Using a dataset of recipes, we analyze macronutrient balance, sodium, sugar, fat content, calories, and the number of steps to build a classification model that identifies patterns in meal preparation time.

## Introduction

Food plays a crucial role in daily life, and cooking time is a key factor in meal planning. According to the USDA Economic Research Service, Americans aged 18 and over spend an average of 37 minutes per day on meal preparation. In today’s fast-paced world, many people prioritize quick, convenient meals, sometimes at the expense of nutritional balance. But do healthier, more balanced meals inherently require longer preparation, or can nutritious meals still be made efficiently?

**This project explores whether a recipe’s nutritional composition and preparation complexity can predict its cooking duration**. Specifically, we investigate how factors such as macronutrient balance, sodium, sugar, fat content, calorie count, and the number of preparation steps influence whether a recipe falls into the short or long cooking time category. Understanding this relationship can help individuals make **better meal-planning choices**, balancing **nutrition, efficiency, and convenience**.

To conduct this analysis, we utilize a dataset from food.com, which contains thousands of recipes along with detailed nutritional profiles, number of steps, ratings and reviews. Originally collected for research on personalized recipe recommendations (Majumder et al.), this dataset serves as the foundation for our analysis and prediction efforts.

The first dataset, `recipes`, consists of 83,782 entries, each representing a unique recipe with 12 recorded attributes, including:

| **Column**         | **Description**     |
| `name`             | Recipe name         |
| `id`             | Recipe ID       |
| `minutes`             | Minutes to prepare recipe         |
| `contributor_id`             | User ID who submitted this recipe        |
| `submitted`             | Date recipe was submitted          |
| `tags`             | Food.com tags for recipe         |
| `nutrition`             | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein(PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value”         |
| `n_steps`             | Number of steps in recipe         |
| `steps`             | Text for recipe steps, in order        |
| `description`             | User-provided description         |
| `ingredients`             | Text for recipe ingredients         |
| `n_ingredients`             | Number of ingredients in recipe        |

The second dataset, `interactions`, contains 731,927 entries, where each row corresponds to a user review for a specific recipe, with various rating-related details such as:


| **Column**         | **Description**     |
| `user_id`             | User ID         |
| `recipe_id`             | Recipe ID       |
| `date`             | Date of interaction         |
| `rating`             | Rating given        |
| `review`             | Review text          |

Since the dataset did not originally contain a measure of nutritional balance, we created a balance score to quantify how closely a recipe aligns with recommended macronutrient distributions of fat, protein, and carbohydrates. Initially, we focused on balance scores as our primary feature. However, through further analysis, we identified additional attributes that improved model performance. We incorporated features such as sodium, sugar, fat content, total calories, and the number of preparation steps, providing a more comprehensive understanding of how nutritional and procedural complexity relate to cooking time.

To ensure consistency across the dataset, we preprocessed the nutritional values and extracted meaningful indicators of recipe complexity. The most relevant features for our classification task include:

- **Balance Score**: A calculated metric that measures how well a recipe aligns with recommended macronutrient distributions.
- **Calories**: Total energy per serving.
- **Sodium & Sugar Content**: Key indicators of recipe composition.
- **Saturated Fat & Total Fat**: Measures of fat composition in the recipe.
- **Number of Steps**: A procedural complexity metric reflecting recipe preparation difficulty.

By building a classification model based on these attributes, we aim to gain insight into how recipe complexity and nutrition correlate with cooking time. This research can help individuals make more informed meal planning decisions by understanding whether certain dietary and procedural characteristics predict longer or shorter preparation times. Additionally, the findings may contribute to future studies exploring the trade-off between nutrition and cooking efficiency, particularly in the context of modern dietary habits.


---

## Cleaning and Exploratory Data Analysis

We conducted the following steps to clean the two dataframes and proceed to the analysis:

1. **Left merged** the recipes and interactions dataset on id and recipe_id in order to match each unique recipe to their corresponding rating and review
2. Dropped irrelevant columns, `Unnamed: 0_x`, `Unnamed: 0_y`, that resulted from the merge
3. Checked the **data types** of all the columns in the dataset to ensure the data types make sense for each column
- | Column         | Description   |
|:---------------|:--------------|
| name           | object        |
| id             | int64         |
| minutes        | int64         |
| contributor_id | int64         |
| submitted      | object        |
| tags           | object        |
| nutrition      | object        |
| n_steps        | int64         |
| steps          | object        |
| description    | object        |
| ingredients    | object        |
| n_ingredients  | int64         |
| user_id        | float64       |
| recipe_id      | float64       |
| date           | object        |
| rating         | float64       |
| review         | object        |

4. For the `rating` column, fill all ratings of **0 with np.nan**
- This is because, generally, the rating goes from 1-5 with 1 indicating the lowest and 5 indicating the highest rating. A value of 0 likely indicates a missing rating instead of an actual rating. But, including a 0 could result in a **distortion when conducting statistical analyses** like mean, median, etc. So, np.nan is a more meaningful placeholder.
5. Added a new column, `average_rating`, that contains the **average rating per recipe**
- Each recipe can have different ratings from different users, so to get a better understanding of the rating of the recipe overall, we can take the average of all the ratings.
6. Converted the nutrition column and extract nutritional values into separate columns of floats
- The `nutrition` column contains strings representing lists, so we converted these strings to a proper JSON format and converted them into a DataFrame with **specific nutrition columns** like `calories (#)`, `total fat (PDV)`, `sugar (PDV)`, `sodium (PDV)`, `protein (PDV)`, `saturated fat (PDV)`, and `carbohydrates (PDV)`. Then we merged these new columns back into the original merged dataframe.
7. Dropped recipes where `calories` are zero
- We did this because we want to calculate the balance score for each recipe, which involves the proportion of `carbohydrates`, `protein`, and `fat` as well as `calories`. When calculating this, we would have calories as the denominator for the proportions, which would result in a **ZeroDivisionError**. The number of rows with calories as 0 was only 102, which is a **small subset** of the entire dataframe, so removing these rows **wouldn’t affect** the analysis.
8. Imputed `fat`, `carbohydrates`, and `protein` with the median for their respective columns
- We used a SimpleImputer with the **median** strategy to fill missing values in the `total_fat`, `carbohydrates`, and `protein` columns, as missing values would interfere with calculating proportions and balance scores, and the **median is robust to outliers**.
9. Calculated the proportions of each macronutrient relative to total calories and add new columns `prop_fat`, `prop_carbs`, `prop_protein` to the dataframe
- We calculated the proportions of fat, carbohydrates, and protein relative to total calories using **standard nutritional guidelines**: fat provides 9 calories per gram, while carbs and protein provide 4. The **specific reference values** (78g fat, 275g carbs, 50g protein) represent **daily recommended intakes**. For each macronutrient, we divided the **percentage of macronutrient** by 100, multiplying it by the **recommended grams of macronutrient** and the **calories per gram of macronutrient**, then dividing by the **total calories** in the recipe.
10. Added `balance_score` column to the dataframe
- The "balance score" measures **how closely the proportions align with an ideal macronutrient distribution** (55% carbs, 25% fat, 20% protein), resulting in balance with higher scores. We calculated the `balance_score` by summing the absolute differences between each macronutrient proportion and its ideal value (carbs: 0.55, fat: 0.25, protein: 0.20), subtracting this sum from 1 to measure how closely the proportions align with the ideal balance.
11. Filtered dataset to exclude extreme outliers in the `minutes` column
- Here, we computed the **interquartile range (IQR)** for the `minutes` column to identify and filter extreme outliers. By defining lower and upper bounds as Q1 − 1.5×IQR and Q3 + 1.5×IQR, we excluded values far outside the typical range. This step is crucial because the minutes column contains  **outliers**, like a maximum of 1,051,200 minutes, which could skew our analysis of how nutrition varies between recipes with short and long cooking times. For our upper bound we got 120 and the lower bound was a negative value, which we considered as 0 minutes, and we filtered our dataset based on these new bounds.
12. Added `cooking_time_category` column to the dataframe
- We categorized recipes as having "short" or "long" cooking times based on whether their cooking time was below or above the **mean of the minutes column** (36.77 minutes). The mean was chosen as the threshold because it provides a central reference point, ensuring a **balanced split between the two categories** for a fair analysis of how nutrition content varies with cooking time. And since we removed the outliers in the previous step, we can ensure that the mean is not affected by outliers.

### Result
Here are the columns of the cleaned dataframe:

| name                                 |     id |   minutes |   calories |   total_fat |   protein |   carbohydrates |   prop_fat |   prop_carbs |   prop_protein |   balance_score | cooking_time_category   |
|:-------------------------------------|-------:|----------:|-----------:|------------:|----------:|----------------:|-----------:|-------------:|---------------:|----------------:|:------------------------|
| 1 brownies in the world    best ever | 333281 |        40 |      138.4 |          10 |         3 |               6 |   0.507225 |     0.238439 |      0.0867052 |        0.317919 | long                    |
| 1 in canada chocolate chip cookies   | 453467 |        45 |      595.1 |          46 |        13 |              26 |   0.542631 |     0.240296 |      0.0873803 |        0.285045 | long                    |
| 412 broccoli casserole               | 306168 |        40 |      194.8 |          20 |        22 |               3 |   0.720739 |     1.2423   |      0.0308008 |       -0.332238 | long                    |
| millionaire pound cake               | 286009 |       120 |      878.3 |          63 |        20 |              39 |   0.503541 |     0.250484 |      0.0888079 |        0.335751 | long                    |
| 2000 meatloaf                        | 475785 |        90 |      267   |          30 |        29 |               2 |   0.788764 |     1.19476  |      0.0149813 |       -0.368539 | long                    |

### Univariate Analysis

To analyze the distribution of cooking times in recipes, we examined a box plot before and after removing outliers. Initially, the data contained extreme values, with some recipes reporting cooking durations close to one million minutes, likely due to misentries. These extreme outliers compressed the rest of the data, making it difficult to interpret meaningful patterns as seen below. To address this, we **removed outliers using the Interquartile Range (IQR) method**, allowing us to focus on a more representative range of cooking times.

<iframe
  src="assets/uni1_fig.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

After removing outliers, the refined box plot below shows that most recipes take between 10 and 60 minutes to prepare, with a median around 30 minutes. The cleaned dataset provides a more accurate view of typical cooking times, making it more reliable for predictive modeling without being skewed by erroneous values.

<iframe
  src="assets/uni2_fig.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


### Bivariate Analysis

To explore the relationship between nutritional composition and cooking time, we analyzed the average proportions of fat, protein, and carbohydrates across short and long cooking time categories.

From the bar chart shown below, we observe that **long-cooking recipes** tend to have a **slightly higher proportion of fat and carbohydrates** compared to short-cooking recipes, while protein content remains relatively similar across both categories. Additionally, the **balance score**, which measures how closely a recipe aligns with recommended macronutrient distributions, is **slightly higher for long-cooking recipes**.

These findings suggest that nutrient composition may play a role in determining cooking time, with longer recipes potentially involving more ingredients or preparation complexity. However, the differences in proportions are not drastic, indicating that additional factors, such as the number of steps and ingredient complexity, may also contribute to cooking duration.

<iframe
  src="assets/bivar_fig.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


### Interesting Aggregates

We analyzed the relationship between cooking time and balance score using a grouped table and a visualization of key statistics over time.
The pivot table shown below provides **aggregated statistics** (mean, median, min, max) for the balance score at different cooking times. This helps us observe how the nutritional balance of recipes fluctuates as preparation time increases.

Here are the first few rows from our aggregated pivot table (has a total of 120 rows for 120 minutes):

|   minutes |   mean_balance_score |   median_balance_score |   min_balance_score |   max_balance_score |
|----------:|---------------------:|-----------------------:|--------------------:|--------------------:|
|         0 |            -0.823702 |             -0.823702  |           -0.823702 |           -0.823702 |
|         1 |             0.118668 |              0.0479904 |           -1.32309  |            0.839363 |
|         2 |             0.181795 |              0.154448  |           -2.52287  |            0.884065 |
|         3 |             0.161351 |              0.122113  |           -1.36223  |            0.876521 |
|         4 |             0.154164 |              0.20384   |           -2.70683  |            0.872263 |


The second visualization below shows the trends of these statistics over cooking time. By examining these aggregates, we observe that while the **mean and median balance** scores remain relatively **stable** across different cooking times, the **minimum balance** score fluctuates more significantly for longer recipes. This suggests that recipes with very low balance scores can sometimes take longer to prepare. Meanwhile, the **maximum balance score** remains high regardless of cooking duration, indicating that highly balanced meals can be achieved across varying preparation times.

<iframe
  src="assets/bal_fig.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

These findings help contextualize our classification model by showing general trends in the data, even if they do not necessarily determine a causal relationship. This analysis supports our broader investigation of how nutrition and preparation complexity relate to cooking time.

---

## Assessment of Missingness

There are three columns in the merged dataset that have a significant amount of missing values. These are `rating`, `description`, and `review`. Due to a significant amount of missingness, we will be analyzing them further.

### NMAR Analysis

We believe that the description column might be Not Missing at Random (NMAR) because people who didn’t add a description might be less engaged with the platform, or maybe the recipe is so simple/self-explanatory that they didn’t feel the need to write one. To determine if it’s Missing At Random (MAR), we would need additional data about the user's activity on the platform like the number of recipes submitted, contribution frequency, information about the recipe like the number of steps, number of ingredients, cooking time (which are provided in our merged dataset), and how familiar the user is with the platform (experienced vs. new users).

### Missingness Dependency

We decided to examine the missingness of `rating` in the merged dataset by testing the dependency of its missingness. We looked at whether the missingness in the `rating` column depends on the column, `n_ingredients`, the number of ingredients in a recipe, or the column `minutes`, the time it took to prepare a recipe.

> #### <span>***Number of Ingredients and Rating***</span>

<ins>**Null Hypothesis**</ins>: The missingness of ratings does not depend on the number of ingredients in the recipe.

<ins>**Alternate Hypothesis**</ins>: The missingness of ratings does depend on the number of ingredients in the recipe.

<ins>**Test Statistic**</ins>: The difference in means in the number of ingredients of the distribution of the group without missing ratings and the distribution of the group without missing ratings.

<ins>**Significance Level**</ins>: 0.05

<iframe
  src="assets/ing_missing.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

We performed a permutation test to evaluate whether the observed difference in means of a column, grouped by a missingness indicator, is statistically significant. By shuffling the `rating` column values multiple times (1000 permutations), we built a null distribution of mean differences and calculated the p-value as the proportion of shuffled differences as extreme as the observed difference.

The observed statistic of **0.1607** is indicated by the red vertical line on the distribution. Since the p_value is 0.0, it is less than 0.05, therefore we r**eject the null hypothesis**. The missingness of `rating` **does depend on** the `n_ingredients`, which is the number of ingredients in a recipe.

> #### <span>***Cooking Minutes and Rating***</span>

<ins>**Null Hypothesis**</ins>: The missingness of ratings does not depend on the cooking minutes in the recipe.

<ins>**Alternate Hypothesis**</ins>: The missingness of ratings does depend on the cooking minutes in the recipe.

<ins>**Test Statistic**</ins>: The difference in means in cooking minutes of the distribution of the group without missing ratings and the distribution of the group without missing ratings.

<ins>**Significance Level**</ins>: 0.05

<iframe
  src="assets/cooking_kde.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
As seen above, outliers in cooking time make it hard to discern the shapes of the two distributions, so we adjust the scale to examine them more closely which can be seen in the plot below.
<iframe
  src="assets/scaled_cooking_kde.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/cooking_emp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Similar to the previous test, we performed another permutation test by shuffling the `rating` column values 1000 times to generate 1000 simulated mean differences between the two distributions based on the test statistic defined above.

The observed statistic is **51.45237** as indicated by the red vertical line on the graph. Since the p-value is **0.12**, which is greater than 0.05, we **fail to reject the null hypothesis**. The missingness of `rating` **does not depend on** the cooking time in `minutes` of the recipe.

---

## Hypothesis Testing

As mentioned previously, we aim to determine whether recipes with **longer cooking times are more nutritionally balanced than those with shorter cooking times**. To investigate this, we conducted a permutation test with the following hypotheses, test statistic, and significance level.

<ins>**Null Hypothesis**</ins>: There is no significant difference in the average balance score between short and long cooking time recipes.

<ins>**Alternate Hypothesis**</ins>: Recipes with long cooking times have significantly higher balance scores than shorter cooking time recipes.

<ins>**Test Statistic**</ins>: The difference in means of the balance scores between long and short cooking time recipes.

<ins>**Significance Level**</ins>: 0.05

The reason for choosing a permutation test is that we lack information about the population distribution and want to assess whether the observed difference in means between long and short cooking times could occur by chance. A permutation test lets us simulate a null distribution by shuffling cooking time labels and recalculating the test statistic, making no strict assumptions about the data.

We hypothesized that longer cooking times may allow for the preparation of more **complex** meals, as they often involve a wider variety of ingredients, which could result in a better balance of macronutrients. For example, stews, soups, or baked dishes with longer cooking times tend to include combinations of vegetables, proteins, and carbohydrates in recommended proportions. On the other hand, shorter cooking time recipes may favor convenience, potentially leading to less nutritionally balanced options like single-ingredient meals. Since we are specifically interested in whether one group tended to outperform the other rather than simply checking for any difference, we have a **directional hypothesis** (higher balance scores for long cooking times), therefore leading to the choice of test statistic, difference in means.

<iframe
  src="assets/hyp_emp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
 
 We conducted a **permutation test** to determine if recipes with long cooking times have significantly higher balance scores than those with short cooking times. First, we calculated the observed difference in mean balance scores between the two groups, which is 0.01567. Then, we simulated the null hypothesis by shuffling the cooking time labels (long or short) across the dataset and recalculating the mean difference 1,000 times to build a null distribution. Finally, we calculated the p-value and got 0.0.

### Conclusion of permutation test

Since the p-value is **0.0,** it is less than the significance level of 0.05, therefore **we reject the null hypothesis**. Based on these results, we have strong evidence to suggest that **recipes with longer cooking times tend to be more nutritionally balanced compared to shorter cooking time recipes**. A **plausible explanation** for this finding could be that recipes with longer cooking times often involve more complex preparation methods or a greater variety of ingredients. However, this conclusion is drawn from a statistical test and should not be interpreted as absolute proof, as there could be other confounding factors influencing this relationship.

---

## Framing a Prediction Problem

This project is formulated as a **binary classification problem**, where the goal is to predict whether a recipe falls into the short or long cooking time category based on its nutritional composition and preparation complexity. Given that cooking time is categorical, classification is a more suitable approach than regression.

To define the response variable, `cooking_time_category`, we used the mean cooking time after removing outliers (36.7 minutes) as the threshold. Recipes with a cooking time below 36.7 minutes were labeled short, while those above were classified as long. The mean was chosen as the threshold because it provides a balanced split of the data, ensuring that both short and long cooking time categories are well-represented. Using the mean rather than an arbitrary cutoff allows for a **data-driven classification** that reflects the natural distribution of cooking times.

For **model evaluation**, we selected **F1-score** as our primary metric. Since some imbalance may exist between short and long cooking time categories, F1-score offers a more reliable assessment than accuracy by considering both precision and recall, preventing the model from favoring one class disproportionately.

At the time of prediction, all features used in the model—such as macronutrient proportions, total calories, sodium, sugar, fat content, and the number of preparation steps—are available before cooking begins. This ensures that the model only relies on information that would be realistically accessible when deciding how long a recipe is likely to take.
By structuring this as a classification problem with a threshold-based approach, this study aims to uncover patterns in cooking duration and provide insights into how nutrition and preparation complexity influence meal planning.

---

## Baseline Model

For our baseline model, we implemented a **Logistic Regression classifier** to predict whether a recipe falls into the short or long cooking time category. The model utilized two features: balance score and calories, both of which underwent preprocessing before being fed into the model.
- **Balance score** is a continuous quantitative feature that measures how closely a recipe’s macronutrient proportions (carbohydrates, fat, and protein) align with an ideal distribution.
- **Calories** is a quantitative feature that we transformed into a nominal (binary) variable using a Binarizer with a 700-calorie threshold. Recipes with ≤ 700 calories were labeled as low-calorie (0), while those > 700 calories were labeled as high-calorie (1).

We applied an **sklearn Pipeline** for preprocessing:
- **Binarizer** converted calories into a categorical variable.
- **StandardScaler** normalized the balance score for consistency.

The dataset was then split into **80% training and 20% testing** to ensure model generalizability.

We assessed model performance using **F1-score** as the primary metric, as it balances precision and recall, making it more informative than accuracy when class imbalance is present. The overall F1-score was **0.235** (rounded), indicating weak classification performance.

The confusion matrix below shows that the model favors predicting short cooking times, correctly classifying 22,500 short recipes but misclassifying 14,790 long recipes as short. This imbalance results in poor recall for long recipes, as only 2,596 were correctly identified.

<iframe
  src="assets/baseline_cf_matrix.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Although the baseline model establishes a benchmark, its **low F1-score** (0.235) suggests that balance score and calories alone do not provide enough information to distinguish short vs. long cooking times effectively. The model is heavily biased toward predicting shorter cooking times, leading to poor recall for long recipes. This highlights the need for additional features—such as recipe complexity indicators (e.g., number of steps, ingredient count)—to enhance predictive performance and improve classification balance.

---

## Final Model

For our final model, we built upon the baseline Logistic Regression classifier by incorporating **additional features** and applying **hyperparameter tuning** to improve predictive performance. The objective remained the same: to classify recipes into short or long cooking times, but with a more refined approach that leveraged both nutritional composition and procedural complexity. 
The final model used `balance_score`, `sodium`, `sugar`, `saturated_fat`, `total_fat`, `calories`, and `n_steps` as features to predict the cooking time category.

**Features**
1. `balance_score`
- The balance score measures how closely a recipe’s macronutrient proportions (carbohydrates, protein, and fat) align with an ideal distribution. The ‘Distribution of Balance Scores by Cooking Time Category’ histogram shows that while short and long-cooking recipes have similar balance score distributions, **long-cooking recipes exhibit slightly greater variation**. This suggests that recipes with diverse macronutrient compositions tend to have longer cooking times, likely due to **increased complexity** and multi-step preparation. Thus, balance score is a **valuable predictor** of cooking duration.

2. `sodium`, `sugar`, `saturated_fat`, `total_fat`
- These macronutrient-related features were **not included** in the calculation of `balance_score`, but provide additional insights into recipe complexity and ingredient composition. Nutritional content often correlates with recipe complexity, as more complex recipes (e.g., baked goods or slow-cooked meals) tend to involve ingredients that contribute to higher levels of sugar, fat, or sodium. These features provide **fine-grained detail** about the recipes, helping us gauge how ingredient composition influences cooking duration.

3. `calories`
- Calories often indicate the **overall composition of a recipe**, with higher-calorie dishes frequently associated with longer cooking times. To enhance predictive power, we applied a Binarizer to classify recipes into "low-calorie" and "high-calorie" categories, **tuning the threshold** to 500 calories instead of 700 calories, as used in the baseline model.
4. `n_steps`
- The number of steps in a recipe is a **strong indicator of complexity**, with multi-step processes often requiring longer preparation times. The ‘Distribution of Number of Steps by Cooking Time Category’ histogram shows that while both short and long-cooking recipes follow a right-skewed distribution, **long-cooking recipes tend to have more steps on average**. This suggests that intricate recipes with more steps, such as casseroles, pastries, or braised dishes, are more likely to fall into the long cooking time category. By incorporating this feature, our model gains a clearer distinction between simpler and more complex recipes, **improving** its ability to accurately predict cooking time categories.

**Preprocessing steps**
1. **StandardScaler** for `sodium`, `sugar`, `saturated_fat`, and `total_fat`:
These features contain potential outliers. So, standardization ensures they are scaled consistently, helping the model avoid being biased toward features with larger ranges.
2. **Binarizer** for `calories`:
The threshold for binarization was tuned to **500** during hyperparameter selection, ensuring that the calorie split aligns with meaningful calorie boundaries in the dataset.
3. **QuantileTransformer** for `balance_score`:
This hyperparameter controls the number of quantile divisions applied to the balance_score. Different numbers of quantiles can better handle varying levels of skew or data granularity, depending on the dataset. By tuning this, we ensure the transformation aligns optimally with the data's characteristics. Transforming balance_score with **200** quantiles addressed any skewness in its distribution, enabling the model to better utilize this continuous feature.
4. **Pass-through** for `n_steps`:
This feature was kept unchanged, as its numeric values naturally reflect recipe complexity.

**Modeling Algorithm**
We chose **Logistic Regression** as our modeling algorithm as our prediction problem is a binary classification. Logistic Regression is also computationally efficient, making it well-suited for this task. To enhance performance, we used **GridSearchCV** with 5-fold cross validation to tune both preprocessing parameters and model hyperparameters. The **F1-score** was used as the scoring metric to ensure balance between precision and recall.

**Hyperparameter Tuning Results**
1. `preprocessor__quantile__n_quantiles`: 200
- A high number of quantiles allowed the model to capture the fine-grained distribution of balance_score, thus handling skews more efficiently.
2. `preprocessor__binarizer__threshold`: 500
- A lower calorie threshold (500) was optimal for distinguishing recipes based on their complexity and duration.
Classifier Hyperparameters:
3. `classifier`__C = 0.01
- This low regularization strength provided better generalization and is ideal to prevent overfitting.
4. `classifier__penalty` = 'l2'
- L2 regularization smoothed the model’s weights helping it perform consistently across both classes.
5. `classifier__solver` = 'liblinear'
- liblinear is ideal for datasets with limited features or smaller datasets.

**Performance Comparison**

Comparing the confusion matrices for the baseline and final model, the final model demonstrates a **significant improvement** over the baseline in its ability to **correctly classify "long" cooking time recipes**. In the baseline model, only 2,596 "long" recipes were correctly predicted, whereas the final model improved this to 7,747, showing a **substantial increase in recall** for the "long" category. Additionally, the number of misclassified "long" recipes (predicted as "short") **dropped** from 14,790 in the baseline to 9,639 in the final model, a marked reduction in false negatives. While the final model **sacrificed some precision** for "short" recipes, with an increase in false positives for "short" labels (from 2,124 to 4,066), this trade-off was necessary to **better balance the classification** of both categories. Overall, the final model significantly reduced the baseline bias toward predicting "short" recipes, achieving a more **balanced and effective performance** for both "short" and "long" cooking times.

<iframe
  src="assets/final_cf_matrix.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The metric **F1 score** of the final model is **0.531**(rounded), which is a **0.3 increase** from the F1 score of the base model. The final model significantly improved over the baseline by incorporating additional features and using hyperparameter tuning to refine both preprocessing and classification. By leveraging meaningful recipe attributes, such as the number of steps and nutritional content, the model effectively captured the nuances of cooking durations. The final F1-score showcases the impact of these enhancements, making the model much more reliable for predicting short versus long cooking times. 

---

## Fairness Analysis
For our fairness analysis, we split the recipes into two groups: **high rating recipes and low rating recipes**. High-rating recipes are defined as recipes with a rating greater than or equal to the median rating of the dataset, while low-rating recipes have a rating below the median. We selected **2.5** as the threshold for ratings since the rating scale ranges from 1 to 5, and 2.5 represents the midpoint, providing an even split for categorizing recipes as either highly or lowly rated.

We evaluate the **precision parity** of the model for these two groups because precision reflects the model's ability to correctly identify cooking time category (short or long) without mislabeling. Precision is particularly **important** to minimize false positives, which could mislead users about cooking times for recipes, impacting their decisions or experiences. For example, if a low-rated recipe is incorrectly predicted as having a long cooking time, users might avoid it unnecessarily.

<ins>**Null Hypothesis**</ins>: Our model is fair. Its precision for high rating and low rating recipes are roughly the same, and any differences are due to random chance.

<ins>**Alternate Hypothesis**</ins>: Our model is unfair. Its precision for low rating recipes is significantly lower than its precision for high rating recipes.

<ins>**Test Statistic**</ins>: The difference in precision between the two groups: (low rating - high rating)

<ins>**Significance Level**</ins>: 0.05

To run the permutation test, first, we split the data into two groups based on the rating threshold of 2.5, categorizing recipes as either "High" or "Low" rated. We calculated the observed precision difference between these two groups **(Low - High)**. To simulate the null hypothesis, we randomly shuffled the rating categories 1,000 times while keeping the true labels and predictions fixed. For each permutation, we recalculated the precision difference to generate a null distribution. Finally, we compared the observed precision difference, which was **0.0401**, to the null distribution to compute the p-value.

<iframe
  src="assets/fair_emp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
With a p-value of **0.003**, which is significantly less than the chosen significance level of 0.05, **we reject the null hypothesis**. This suggests that there is evidence of a **disparity in precision**, where the model performs **less effectively** for low-rated recipes compared to high-rated ones.

---

## Conclusion

In this project, we developed a classification model to predict whether a recipe falls into the short or long cooking time category, using **nutritional composition and procedural complexity** as key predictors. Our baseline **Logistic Regression model**, relying only on `balance_score` and `calories`, struggled to distinguish between the two categories, achieving an **F1-score of 0.235**, with poor recall for long recipes.

To improve performance, we incorporated `sodium`, `sugar`, `saturated_fat`, `total_fat`, and `n_steps`, capturing both ingredient composition and recipe complexity. **Exploratory data analysis** supported these choices, showing that `n_steps` strongly correlated with longer cooking times, while sodium, sugar, and fat content acted as indicators of ingredient-driven complexity. **Preprocessing steps**, including **Quantile Transformation** for `balance_score`, **Standard Scaling** for macronutrients, and **Binarization** for `calories` with an optimized 500-calorie threshold, helped normalize the data. **GridSearchCV with 5-fold cross-validation** further optimized model performance.

These refinements significantly improved classification, raising the **final F1-score to 0.531**, more than **double** the baseline model’s performance. The improved recall for long-cooking recipes suggests that recipes with greater macronutrient variation, higher sodium and fat content, and more preparation steps are more likely to require longer cooking times. This confirms that **both nutritional composition and procedural complexity are key factors influencing cooking duration**.

Ultimately, our model provides **meaningful insight into how a recipe’s attributes relate to cooking time**, highlighting patterns that could inform recipe planning, meal recommendations, or even automated cooking assistance. This study demonstrates the power of integrating both nutritional and procedural data to make informed predictions about cooking duration, reinforcing the idea that complexity—whether in ingredients or steps—is a strong determinant of preparation time.

---
