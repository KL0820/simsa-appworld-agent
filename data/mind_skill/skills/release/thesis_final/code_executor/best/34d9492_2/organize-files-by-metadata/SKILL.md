---
name: organize-files-by-metadata
description: Systematically organizes files into new directory structures based on their extracted metadata, such as creation dates or other attributes.
---
## Overview
This pattern addresses tasks requiring the systematic arrangement of files within a file system. It involves an initial phase of discovering files and their relevant metadata, followed by a classification step where files are assigned to target locations based on specific criteria derived from their metadata. Finally, the files are moved or copied to their designated new locations, often involving the creation of new directories. This strategy focuses on iterating through files, enriching their data, and performing conditional actions to create a structured directory hierarchy.

## When to Apply
- Organize files into subdirectories based on their attributes (e.g., date, type, vacation spot).
- Process a collection of files and sort them.
- Create a structured directory hierarchy from existing flat files.
- Classify files by creation month and year.

## Procedure
1.  **Identify Source**: Define the base directory containing the files to be organized.
2.  **List Files**: Recursively list all files within the base directory.
3.  **Retrieve & Enrich Metadata**: For each identified file, retrieve additional detailed metadata that may not be available in the initial listing (e.g., creation date, size, type). This often requires individual calls per file.
4.  **Parse & Transform Metadata**: Parse and transform the raw metadata into a structured format suitable for comparison and categorization (e.g., converting date strings to date objects and extracting components like year and month).
5.  **Define & Ensure Target Directories**: Define the target subdirectories where files will be moved or copied. For each category, ensure the corresponding destination directory exists, creating it if necessary, and handling pre-existing directories gracefully.
6.  **Categorize Files**: Iterate through the collected file metadata. For each file, apply conditional logic based on its transformed metadata to determine its appropriate target subdirectory.
7.  **Construct Destination Path**: Construct the full destination path for the file, combining the base path, the dynamically determined category-specific subdirectory, and preserving its original filename.
8.  **Perform File Operation**: Perform the file system operation (move or copy) to place the file in its designated location.

## Key Patterns
-   **Two-Phase Data Retrieval / Multi-step Data Enrichment**: First, retrieve a list of file paths, then make individual calls for each path to fetch detailed metadata.
-   **Metadata-Driven Classification & Data Transformation**: Use conditional statements (if/elif/else) to categorize files into different destination groups based on parsed and transformed metadata attributes (e.g., date components, string patterns). Raw data is parsed and transformed into structured components that enable conditional logic and grouping.
-   **Idempotent Directory Management**: Create target directories using an option that prevents errors if the directory already exists, or rely on file system operations that implicitly create necessary parent directories, ensuring the operation can be safely re-run.
-   **Dynamic Destination Path Construction**: Carefully combine base paths, dynamically determined subdirectories, and original filenames to form complete and correct destination paths.
-   **Robust API Response Handling**: Check API responses for expected types and keys (e.g., `isinstance(response, dict)`, `response.get('key')`) to gracefully handle empty results, errors, or variations in success payloads.
-   **Prior Variable Utilization**: Access and process data collected and stored in variables from preceding milestones.

## Common Pitfalls
-   **Incomplete Metadata**: Assuming a single API call provides all necessary metadata for categorization, leading to incomplete data.
-   **Untransformed Metadata**: Failing to convert raw metadata into a structured, comparable format, making categorization logic difficult or impossible.
-   **Directory Management Errors**: Not creating destination directories before attempting to move files into them, or failing to handle existing directories, or not accounting for their existence.
-   **Filename/Path Issues**: Not preserving the original filename when moving or copying files, or incorrectly deriving/preserving original item names, leading to lost or overwritten data.
-   **Misclassification**: Incorrectly parsing or comparing date/time strings or other metadata, leading to misclassification of files, or applying move operations to all items without proper categorization.
-   **API/Error Handling**: Failing to handle cases where directory listing or file detail retrieval returns an error or an empty result, or ignoring potential API errors or unexpected response formats.
-   **Flat Structure Assumption**: Assuming a flat directory structure when files might be nested, requiring recursive listing.