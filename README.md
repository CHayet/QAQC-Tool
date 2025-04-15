## QA/QC Tool 

<img width="615" alt="image" src="https://github.com/user-attachments/assets/7ee000ad-b9c6-4a53-9506-d87d4fdee09f" />


### Project Overview
The QA/QC Tool is a data validation and quality assurance application built with Streamlit for an interactive user experience. This tool is designed to streamline the process of validating data quality by performing a series of key checks on files, including duplicates, outliers, column name consistency, inter-file consistency, and a general data summary.

This tool supports CSV and Excel file formats. The primary goal is to ensure the integrity and consistency of files, identify discrepancies, and help to improve the reliability of the data.

### General QA/QC Process 
The QA/QC process includes a series of structured checks to ensure data integrity, consistency, and usability. These checks typically involve:\
•	Verifying the consistency of file tabs and names\
•	Ensuring that totals match the sum of individual items\
•	Checking that data is within expected ranges\
•	Identifying any missing data\
•	Confirming that the data is properly formatted\
•	Comparing current data with previous years to spot unexpected differences\
•	Reviewing documentation and README files

### Automated Process
The automated QA/QC tool performs a series of checks to ensure data quality. Once a file is uploaded, the system runs through predefined checks that address key data quality issues, such as missing values, outliers, and inconsistencies. The results are displayed in the Streamlit interface, and are also logged in a txt report for further review and documentation.\
Key Features:\
• File Upload: Users can upload CSV or Excel files to start the Qa/QC process.\
• Column Consistency Check: Compares column names between files (e.g., Current year vs last year) to check for any mismatches.\
• Inter-file Consistency Check: Verify that the sum of a metric reported across multiple files remains consistent across all relevant files.\
• Outlier Detection: Detects outliers using log transformation and the adjusted interquartile range (IQR) method.\
• Duplicates Check: Identifies duplicate rows in the file or in selected columns.\
• Data Summary: Generates a summary of the data, including descriptive statistics, missing values, and whitespace issues.\
• Report Generation: Generates a detailed text-based report summarizing all checks performed, which can be downloaded.

### Benefits of Automation
The automated process significantly reduces the manual effort involved in data validation by performing checks swiftly and consistently. Some of the main benefits of automation include:\
• Increased Efficiency: The tool automates repetitive tasks, saving time and reducing the risk of human error.\
• Consistency: Automated checks ensure that the process is carried out uniformly each time, eliminating the variation that can occur in manual reviews.\
• Scalability: As data volumes grow, the tool can scale to handle larger datasets and multiple files without compromising performance.\
• Real-Time Results: Users receive immediate feedback on the quality of their data, allowing for quick adjustments and decisions.

### Installation 
To use the QA/QC Tool, you will need to install the necessary Python libraries. The following steps outline how to set up the tool:

**Prerequisites**\
• Python 3.x (recommended version: Python 3.7 or later)\
• Streamlit for the interactive web interface\
• Pandas and NumPy for data manipulation\
• openpyxl (for Excel file support)

**Installation Steps**
1. Install the required dependencies: pip install pandas numpy streamlit openpyxl deepchecks.
2. Download or clone the repository containing this tool to your local machine.
3. Run the Streamlit application: streamlit run qaqc_tool.py.
4. The app will open in your default web browser where you can upload your dataset and begin the quality checks.
