***Machine Learning Dataset Log***



**Date:**23-10-2026



The goal of this step is to prepare the final datasets for model training. This involves separating the input features (X) from the target variable (y), handling any missing values, splitting the data into training and testing sets, and ensuring no data leakage occurs.



**What was done:**

**1. Loaded Reduced Datasets:** Loaded the feature-selected CSVs from previous step(`selected\_features\_Tl.csv`, `selected\_features\_Tx.csv`, `selected\_features\_Tg.csv`).

**2. Separated X and y:**

&#x20;  - `X` (Input Features) was created by dropping the `Composition` string and the target column.

&#x20;  - `y` (Target Variable) was isolated as the single temperature column (e.g., `Tl\_\[K]`).

&#x20;  - Note: The target variable shape is 1-Dimensional (e.g., `(823,)`) because it only contains a single list of values to predict.

**3. Removed Missing Values:** Dropped any rows where the target value was missing to prevent errors during training.

**4. Train-Test Split:** Split the data using an 80/20 ratio with `random\_state=42` for reproducibility.

**5. Saved Final Datasets:** Saved the `X\_train`, `X\_test`, `y\_train`, and `y\_test` combined back into CSV files inside the `data/ml\_datasets/` folder.



**Results \& Dataset Shapes**

Based on the terminal output, here are the final shapes for each target:



Tl (Liquidus Temperature)

\- Input Features (X): 823 rows, 4 columns

\- Target (y): 823 rows

**- Train Set:** 658 records

**- Test Set:** 165 records



Tx (Crystallization Temperature)

\- Input Features (X): 856 rows, 4 columns

\- Target (y): 856 rows

**- Train Set:** 684 records

**- Test Set:** 172 records



Tg (Glass Transition Temperature)

\- Input Features (X): 883 rows, 4 columns

\- Target (y): 883 rows

**- Train Set:** 706 records

**- Test Set:** 177 records



**Note on Classification (Dmax)**

Our dataset only contains positive examples (Dmax > 0). Because there is no negative class (Dmax = 0), it is mathematically impossible to train a binary classification model. Therefore, we proceeded directly to the regression tasks (Tl, Tx, Tg) which are fully populated.



**5. Saved Files**

The following files were generated and saved in `data/ml\_datasets/`:

\- `train\_Tl.csv` and `test\_Tl.csv`

\- `train\_Tx.csv` and `test\_Tx.csv`

\- `train\_Tg.csv` and `test\_Tg.csv`

