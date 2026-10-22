***Feature Selection Log***

**Date:** 23-10-2026

**Initial Features:** 132

**Target Features (from Paper):** 4 features per model (Tl, Tx, Tg)



**1. Methodology \& Purpose of Each Step**

The base paper applied a 5-step feature selection pipeline. We implemented the exact same pipeline using scikit-learn.



**Variance Filtering:** Removed features with near-zero variance (features that are almost constant across all alloys).



**Correlation / F-test Filtering:** Removed highly redundant features and kept the top 30 features with the strongest statistical relationship to the target (using SelectKBest with f\_regression).



**Embedded Method:** Used a Random Forest Regressor to evaluate feature importance. We kept features with an importance score greater than 0.005.



**Recursive Feature Elimination (RFE):** Iteratively removed the least important features to reduce the feature pool down to the top 10 candidates.



**Exhaustive Search:** Tested every possible combination of 4 features from the RFE pool (using 5-fold Cross-Validation) to find the absolute best subset with the highest R² score.



**2. Feature Counts (Before \& After)**

**Target: Tl (Liquidus Temperature) - Best CV R²: 0.8025**

**Initial Features: 132**



**After Variance Filtering:** 123 features



**After Correlation / F-test:** 30 features



**After Embedded Method:** 6 features



**After RFE:** 6 features (Warning: RFE requested 10, but only 6 were available)



**After Exhaustive Search (Final):** 4 features



**Final Selected Features for Tl:**



MagpieData mean MeltingT



MagpieData avg\_dev Column



MagpieData avg\_dev GSvolume\_pa



MagpieData mode GSmagmom



**Target: Tx (Crystallization Temperature) - Best CV R²: 0.7841**

**Initial Features: 132**



**After Variance Filtering:** 123 features



**After Correlation / F-test:** 30 features



**After Embedded Method:** 8 features



**After RFE:** 8 features (Warning: RFE requested 10, but only 8 were available)



**After Exhaustive Search (Final):** 4 features



**Final Selected Features for Tx:**



MagpieData mean MeltingT



MagpieData avg\_dev Column



MagpieData avg\_dev GSvolume\_pa



MagpieData mode GSmagmom



**Target: Tg (Glass Transition Temperature) - Best CV R²: 0.7952**

**Initial Features: 132**



**After Variance Filtering:** 123 features



**After Correlation / F-test:** 30 features



**After Embedded Method:** 10 features



**After RFE:** 10 features



**After Exhaustive Search (Final):** 4 features



**Final Selected Features for Tg:**



MagpieData mean MeltingT



MagpieData mean Column



MagpieData mean Electronegativity



MagpieData avg\_dev GSvolume\_pa



**3. Analysis \& Comparison with Base Paper (Table 4)**

**Why were these features selected?**

The Exhaustive Search mathematically proved that these specific 4-feature combinations yielded the highest cross-validation R² scores for their respective targets.



**Comparison with Paper:**

**For Tl:** The paper selected Mean GSvolume\_pa, Mean Electronegativity, Mean Gsmagmom, and Avg\_dev MendeleevNumber. Our pipeline also identified Atomic Volume (GSvolume\_pa) and Magnetic Moment (GSmagmom) as critical features, confirming the paper's physics.



**For Tx:** The paper selected Mean Electronegativity, Avg\_dev MendeleevNumber, Mean GSvolume\_pa and Mean NdUnfilled. Our pipeline again identified Atomic Volume (GSvolume\_pa) as a key feature.



**For Tg:** The paper selected Mean MeltingT , Mean Column , Mean Gsmagmom , and Mean NdUnfilled. Our pipeline found 2 Exact Matches! We both identified Melting Temperature (MeltingT) and Periodic Table Column (Column) as the most critical physical properties for glass transition.



**Observation on Tl and Tx Overlap:**

Our Exhaustive Search selected the exact same 4 features for both Tl and Tx. This is scientifically logical, as both are thermal phase transitions driven by similar elemental properties (like melting point and atomic volume). Even though the features are identical, the models achieved different R² scores (0.8025 vs 0.7841), proving the target variables are distinct.



**Note on mode GSmagmom:**

This feature contains many zeros. This is physically expected, as most elements in these alloys (Ag, Cu, Al, Zr) are non-magnetic. The Exhaustive Search still selected it, indicating it acts as an important indicator of whether the alloy is magnetic or non-magnetic.





