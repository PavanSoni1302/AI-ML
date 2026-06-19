# Day 15 Report — AI/ML Developer Track

## Unsupervised Learning: User Segmentation using K-Means

## Technical Summary

Today I implemented K-Means Clustering to identify hidden user segments without using predefined labels. Unlike supervised learning, the model grouped users based on similarity in their behavior using distance-based clustering.

## Problem Approach

The objective was to segment users based on their activity patterns. Since no labels were provided, I used K-Means clustering to automatically detect groups in the dataset.

To ensure meaningful clustering:

* I applied **StandardScaler** to normalize the data
* Used the **Elbow Method** to determine the optimal number of clusters (K)
* Selected **K = 5** based on the elbow point

## Implementation Details

* Generated a dataset representing user behavior patterns
* Applied scaling to avoid bias in distance calculations
* Implemented K-Means algorithm with `init='k-means++'` for better centroid initialization
* Visualized clusters using scatter plots
* Marked centroids to show cluster centers

## Key Observation

The Elbow Method showed a clear bend at K = 5, indicating the optimal number of clusters.
The clusters formed were distinct and well-separated, showing that the model successfully grouped similar users.

## Experiment Insight (Initialization Test)

I tested K-Means with different initialization methods:

* `random` initialization resulted in varying cluster centers
* `k-means++` produced more stable and consistent results

This shows that initialization plays a critical role in clustering performance.

## Conceptual Reflection

Clustering allows us to discover hidden user segments based on behavior.
For example, if a group of users prefers late-night activities and high-intensity engagement, the system can recommend similar events, users, or content to them.

This improves personalization and enhances the overall user experience.

## Key Learning

K-Means is highly sensitive to:

* Feature scaling
* Initialization method
* Choice of K

Proper preprocessing and parameter selection are essential for meaningful clustering results.
