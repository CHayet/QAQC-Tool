import pandas as pd
import numpy as np
import os
import streamlit as st
import warnings
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import MixedDataTypes
from deepchecks.tabular.checks import SpecialCharacters

class QAQC:
    def __init__(self):
        # Streamlit for file uploads
        logo_url = "https://streamlit.io/images/brand/streamlit-mark-color.png"

        # Custom header with title and logo inline
        st.markdown(f"""
            <h1 style='display: flex; align-items: center; gap: 10px;'>
                Streamlit QA/QC Tool
                <img src='{logo_url}' width='60' style='margin-top: 3px;'/>
            </h1>
        """, unsafe_allow_html=True)
        
        # suppress specific warning
        warnings.filterwarnings("ignore", message="Cannot parse header or footer so it will be ignored")
        
        self.dataset1_file = st.file_uploader("Upload the file (CSV or Excel) to start the QA/QC review", type=["csv", "xlsx"])

        self.df1 = None  # initialize df1
        self.dataset1_name = ""
        self.report = []  # list to collect report content
        self.check_count = 0  # counter to make keys unique

    # Load the Excel or CSV data
    def load_data(self, file):
        try:
            if file.name.endswith(".xlsx"):
                excel_file = pd.ExcelFile(file)
                if len(excel_file.sheet_names) > 1:
                    selected_sheet = st.selectbox("Multiple sheets detected. Please select a sheet.", excel_file.sheet_names)
                    return pd.read_excel(file, sheet_name=selected_sheet) 
                else:
                    return pd.read_excel(file)
            elif file.name.endswith(".csv"):
                return pd.read_csv(file)
        except Exception as e:
            st.error(f"Error loading file: {e}")         
            return None
                
    
    # check if column names match between df1 and an uploaded file for consistency check   
    def check_column_names_match(self, additional_file):
        
        if additional_file.name.endswith('.xlsx') or additional_file.name.endswith('.xls'):
            excel_file = pd.ExcelFile(additional_file)
            if len(excel_file.sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "Multiple sheets detected. Please select a sheet for comparison.",
                    excel_file.sheet_names
                )
                additional_df = pd.read_excel(excel_file, sheet_name=selected_sheet)
            else:
                additional_df = pd.read_excel(excel_file)
        else:
            additional_df = pd.read_csv(additional_file)
        
        self.report.append("\n"+ "="*80)
        self.report.append(f"*** Column Name Consistency Check: {self.dataset1_name} vs {os.path.splitext(os.path.basename(additional_file.name))[0]} ***\n")
                 
        columns_df1 = set(self.df1.columns)
        columns_df2 = set(additional_df.columns)

        # find the columns that are only in one dataset
        columns_in_df1_not_in_df2 = columns_df1 - columns_df2
        columns_in_df2_not_in_df1 = columns_df2 - columns_df1

        if columns_in_df1_not_in_df2 or columns_in_df2_not_in_df1:
            self.report.append("Column names in the datasets do not match:")
            if columns_in_df1_not_in_df2:
                self.report.append(f"Columns in {self.dataset1_name} but not in the uploaded file: {sorted(columns_in_df1_not_in_df2)}\n")
            if columns_in_df2_not_in_df1:
                self.report.append(f"Columns in uploaded file but not in {self.dataset1_name}: {sorted(columns_in_df2_not_in_df1)}\n")
            st.error(f" Column names in the datasets do not match:")
            if columns_in_df1_not_in_df2:
                st.write(f"Columns in {self.dataset1_name} but not in the uploaded file: {sorted(columns_in_df1_not_in_df2)}")
            if columns_in_df2_not_in_df1:
                st.write(f"Columns in uploaded file but not in {self.dataset1_name}: {sorted(columns_in_df2_not_in_df1)}")
        else:
            self.report.append("Column names match between the datasets.")
            st.success("Column names match between the datasets.")
            
    # function to prompt the user to select columns for inter-file check using Streamlit
    def get_column_from_user(self, df1, df2, dataset1_name, dataset2_name):
        st.write(f"\nAvailable columns in {dataset1_name}:")
        column_options1 = df1.columns.tolist()
        column_index1 = st.selectbox(f"Select column from {dataset1_name} to check for consistency", column_options1, key=f"column_select_1_{self.check_count}")

        st.write(f"\nAvailable columns in {dataset2_name}:")
        column_options2 = df2.columns.tolist()
        column_index2 = st.selectbox(f"Select column from {dataset2_name} to check for consistency", column_options2, key=f"column_select_2_{self.check_count}")

        st.write(f"Selected column from {dataset1_name}: {column_index1}")
        st.write(f"Selected column from {dataset2_name}: {column_index2}")

        return column_index1, column_index2

    # perform inter-file consistency check
    def inter_file_check(self):
        self.report.append("\n" + "=" * 80)
        self.report.append(f"*** Inter File Consistency Check ***\n")        
        st.write("Performing inter-file consistency check.")

        # increment the check count to ensure uniqueness
        self.check_count += 1

        # allow the user to upload an additional file for inter-file check
        additional_file = st.file_uploader("Select a file to compare with the uploaded dataset", type=["csv", "xlsx"], key=f"additional_file_{self.check_count}")

        if additional_file is not None:
                      
            # read the uploaded file based on its type: csv or excel 
            if additional_file.name.endswith('.csv'):
                additional_df = pd.read_csv(additional_file)
                additional_dataset_name = os.path.splitext(os.path.basename(additional_file.name))[0]
            elif additional_file.name.endswith('.xlsx'):
                excel_file = pd.ExcelFile(additional_file)
                additional_dataset_name = os.path.splitext(os.path.basename(additional_file.name))[0]

                # if the excel file has multiple sheets,let the user to select one
                if len(excel_file.sheet_names) > 1:
                    selected_sheet = st.selectbox("Multiple sheets detected. Please select a sheet for comparison.", excel_file.sheet_names)
                    additional_df = pd.read_excel(additional_file, sheet_name=selected_sheet)
                else:
                    additional_df = pd.read_excel(additional_file)


            # get the names of the previously uploaded datasets
            dataset1_name = os.path.splitext(os.path.basename(self.dataset1_file.name))[0]
            selected_df=self.df1
           

            column_to_check_df1, column_to_check_df2 = self.get_column_from_user(selected_df, additional_df, dataset1_name, additional_dataset_name)
            
            
            # check for first column: only convert if the first value is not a string and data type is not already numeric
            def is_first_value_text(column):
                # Check if the first value is not NaN and is a string
                return isinstance(column.iloc[0], str) and pd.notna(column.iloc[0])

            # check if the first value in the column is numeric or text and perform conversion accordingly
            if selected_df[column_to_check_df1].dtype not in ['float64', 'int64'] and not is_first_value_text(selected_df[column_to_check_df1]):
                selected_df[column_to_check_df1] = pd.to_numeric(selected_df[column_to_check_df1], errors='coerce')

            if additional_df[column_to_check_df2].dtype not in ['float64', 'int64'] and not is_first_value_text(additional_df[column_to_check_df2]):
                additional_df[column_to_check_df2] = pd.to_numeric(additional_df[column_to_check_df2], errors='coerce')
                
            # check if the selected columns are numeric 
            if selected_df[column_to_check_df1].dtype not in ['float64', 'int64']:
                st.error(f"Column '{column_to_check_df1}' in {dataset1_name} is not numeric. Please select a numeric column for comparison.")
                return  # exit the function if the column is not numeric

            if additional_df[column_to_check_df2].dtype not in ['float64', 'int64']:
                st.error(f"Column '{column_to_check_df2}' in {additional_dataset_name} is not numeric. Please select a numeric column for comparison.")
                return  # exit the function if the column is not numeric
                     
            #fill NaN values with 0
            selected_df[column_to_check_df1] = selected_df[column_to_check_df1].fillna(0)
            additional_df[column_to_check_df2] = additional_df[column_to_check_df2].fillna(0)
            
            #calculate the sum of the selected columns
            sum_selected_df = selected_df[column_to_check_df1].sum()
            sum_additional_df = additional_df[column_to_check_df2].sum()

            tolerance = 1e-5
            if abs(sum_selected_df - sum_additional_df) <= tolerance:
                self.report.append(f"Inter-file check for column '{column_to_check_df1}': The sums match between {dataset1_name} and {additional_dataset_name}.")
                st.success(f"Inter-file check for column '{column_to_check_df1}': The sums match between {dataset1_name} and {additional_dataset_name}.")
            else:
                self.report.append(f"Inter-file check for column '{column_to_check_df1}': The sums do not match.")
                self.report.append(f"1. Sum of {column_to_check_df1} in {dataset1_name}: {sum_selected_df}")
                self.report.append(f"2. Sum of {column_to_check_df2} in {additional_dataset_name}: {sum_additional_df}")
                st.error(f"Inter-file check for column '{column_to_check_df1}': The sums do not match.")
                st.write(f"1. Sum of {column_to_check_df1} in {dataset1_name}: {sum_selected_df}")
                st.write(f"2. Sum of {column_to_check_df2} in {additional_dataset_name}: {sum_additional_df}")
           
        else:
            st.warning("Please upload the file to compare with the uploaded datasets.")    
   
     # check for outliers in the dataset using log transformation and adjusted IQR
    def check_outliers_log_iqr(self, df, dataset_name, columns_to_check=None):
        if df is not None and not df.empty:  # check if file is not None and not empty
            self.report.append("\n"+ "="*80)
            self.report.append(f"*** Outliers Check ***\n")
                   
            if columns_to_check is None:
                columns_to_check = df.select_dtypes(include=["number"]).columns.tolist()
                
            # Check for non-numeric columns in columns_to_check and display a message if any are found
            for column in columns_to_check:
                # check if the column is numeric or can be safely converted to numeric
                if df[column].dtype not in ['float64', 'int64']:
                    # convert the column to numeric only if the first value is numeric
                    if not isinstance(df[column].iloc[0], str) or pd.isna(df[column].iloc[0]):
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    else:
                        #st.warning(f"Column '{column}' in {dataset_name} is not numeric and cannot be converted. Skipping outlier check for this column.")
                        continue  # Skip this column if it's not numeric and cannot be converted
                        
            non_numeric_columns = [col for col in columns_to_check if not np.issubdtype(df[col].dtype, np.number)]
            if non_numeric_columns:
                st.warning(f"Please select only numeric columns for outlier checking. The following columns are non-numeric: {', '.join(non_numeric_columns)}")
                return  # exit the function if non-numeric columns are selected

            for column in columns_to_check:
                cleaned_data = df[column].dropna()
                cleaned_data = [x for x in cleaned_data if x > 0]  # remove non-positive values
                if len(cleaned_data) > 0:
                    log_transformed_data = np.log1p(cleaned_data)
                    Q1 = np.percentile(log_transformed_data, 25)
                    Q3 = np.percentile(log_transformed_data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR

                    # identifying the outlier indices
                    outliers_log_indices = [i for i, x in enumerate(log_transformed_data) if x < lower_bound or x > upper_bound]
                    outliers_original = [np.expm1(log_transformed_data[i]) for i in outliers_log_indices]
                
                    tolerance = 1e-5
                    outlier_indices = []
                
                    for value in outliers_original:
                        # check for values within the tolerance range
                        matching_indices = df[column][np.abs(df[column] - value) <= tolerance].index
                        if not matching_indices.empty:  # check if any matching indices are found                            
                            outlier_indices.append(matching_indices[0] + 1)  # adding 1 for 1-based Excel indexing
                        else:
                            outlier_indices.append(None)  # if no match, append None or handle as needed

                    # reporting the outliers including the index
                    if outliers_original:
                        self.report.append(f"Outliers detected in column '{column}':")
                        st.error(f"Outliers detected in column '{column}':")
                        for idx, outlier in zip(outlier_indices, outliers_original):
                            formatted_outlier = f"{outlier:.2f}"  # format to 2 decimal places
                            self.report.append(f"  Index {idx}: {formatted_outlier}")
                            st.write(f"  Index {idx}: {formatted_outlier}") 
                    else:
                        self.report.append(f"No outliers detected in column '{column}'.")
                        st.success(f"No outliers detected in column '{column}'.")
                else:
                    self.report.append(f"No valid data to check for outliers in column '{column}'.")
                    st.write(f"No valid data to check for outliers in column '{column}'.")
    
        else:
            self.report.append(f"{dataset_name} - DataFrame is empty. No outlier check performed.")
            st.write(f"{dataset_name} - DataFrame is empty. No outlier check performed.")          
   
     # duplicates check
    def check_dupes(self, df, cols, dataset_name):
        dupe_cols = cols
        if dupe_cols == []:
            dupe_cols = df.columns.tolist()
        dupes_count = df.duplicated(subset = dupe_cols).sum()
        self.report.append("\n" + "=" * 80)
        self.report.append(f"\n*** Duplicates Check ***\n")
        #st.subheader(f"Duplicates Check")
        st.write(f"Number of duplicate records: {dupes_count}")
        if dupes_count > 0:
            duplicate_indices = df[df.duplicated(subset = dupe_cols, keep='first')].index.tolist()
            excel_indices = [index + 1 for index in duplicate_indices]
            self.report.append(f"Duplicate record indices (Excel): {excel_indices}")
            st.write(f"Duplicate record indices (Excel): {excel_indices}")
        else:
            self.report.append("No duplicates found.")
            st.success("No duplicates found.")
            

    def handle_timestamp_columns(self, df):
        """Convert any Timestamp columns to strings to ensure compatibility with pyarrow."""
        for col in df.select_dtypes(include=['datetime64[ns]', 'object']):
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # convert datetime to string format 
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        return df

    def check_summary(self, df, dataset_name):
        
        st.subheader(f"Data Summary")
        
        # handle any Timestamp colums
        df = self.handle_timestamp_columns(df)
        
        keywords = ['nitd id', 'uace', 'id', 'code', 'zip'] # apply changes to this list based on columns that are of object data type but contain numeric values, due to differences in how Python and Excel detect and interpret data types in some cases.

        # convert specific columns to string if their names contain certain keywords (case-insensitive)
        for col in df.columns:
            if any(keyword in col.lower() for keyword in keywords):
                df[col] = df[col].astype(str)
                #print(f"Column '{col}' in Current File converted to string.") --- uncomment for debugging 
    
        # data Description
        st.write("Data Description:")
        description = df.describe()
        st.write(description)  # display in the app
    
        # missing values        
        missing_values = df.isnull().sum()
        missing_columns = missing_values[missing_values > 0].sort_values(ascending=False)
    
        if not missing_columns.empty:
            st.write("🚨 Columns with Missing Values:")
            missing_df = missing_columns.reset_index()
            missing_df.columns = ["Column Name", "# of Missing Values"]
            st.dataframe(missing_df)
        else:
            st.write("✅ No missing values in any columns.")
            self.report.append("No missing values in any columns.\n")
            
        whitespace = {}

        for col in df.select_dtypes(include=["object"]):  # Only check object (string) columns
            whitespace_rows = df[col].apply(lambda x: isinstance(x, str) and x != x.strip())
            count = whitespace_rows.sum()
            if count > 0:
                whitespace[col] = count  # Save the number of affected rows

        if whitespace:
            st.write("🚨 Columns with Whitespace Issues:")
    
            message = ""
            for col, count in whitespace.items():
                message += f"{col}: {count} rows with leading/trailing whitespace\n"
    
            st.text_area("Detected whitespace in these columns:", value=message.strip(), height=200)
            self.report.append(f"*** Whitespace Check *** \n")
            self.report.append(f"Whitespace Check detected in the following columns:\n{message}\n")
        else:
            st.write("✅ No whitespace issues detected in any columns.")
    #this function checks for mixed data types in all columns
    def check_mixed_data(self, df):
       
        # infer categorical features: strictly columns with dtype object, bool, or category
        cat_features = [col for col in df.columns if df[col].dtype in ['object', 'bool', 'category']]
        dataset = Dataset(df, cat_features=cat_features)
        
        # r the MixedDataTypes check
        check = MixedDataTypes()
        result = check.run(dataset)
    
        # access mixed data directly via 'value' method
        mixed_data = result.value  # this should contain the dictionary with mixed data
    
        # if mixed data is found, convert it to a DataFrame and display
        if mixed_data:
            # create a DataFrame based on the mixed_data structure
            mixed_data_df = pd.DataFrame.from_dict(mixed_data, orient='index')
        
            # check if mixed_data_df has data (rows)
            if mixed_data_df.empty:
                #st.write("No mixed data found in the dataset.")
                return None
        
            # Rename columns to match the desired format
            mixed_data_df.columns = ['strings', 'numbers', 'strings_examples', 'numbers_examples']
        
            return mixed_data_df
        else:
            #st.write("No mixed data found.")
            return None
        
    #The SpecialCharacters check search in column[s] for values that contains only special characters.
    def check_Special_Characters(self, df):
       
        # infer categorical features: strictly columns with dtype object, bool, or category
        cat_features = [col for col in df.columns if df[col].dtype in ['object', 'bool', 'category']]
        
        dataset = Dataset(df,cat_features=cat_features )        
        
        result = SpecialCharacters().run(dataset)
    
        # access Special_Characters data directly via 'value' method
        special_chars_data = result.value  # This should contain the dictionary with mixed data
    
        # if Special_Characters is found, convert it to a DataFrame and display
        if special_chars_data:
            # create a DataFrame based on the mixed_data structure
            special_chars_df = pd.DataFrame.from_dict(special_chars_data, orient='index')
            special_chars_df.columns = ["% of Special Characters Detected"]
            filtered_df = special_chars_df[special_chars_df["% of Special Characters Detected"] > 0]
        
            # check if mixed_data_df has data (rows)
            if filtered_df.empty:
                #st.write("No special Character found in the dataset.")
                return None
        
            return filtered_df
        else:
            #st.write("No special Character found.")
            return None

    # Generate final report
    def generate_report(self):
        report_content = f"# QA/QC Check Results for {self.dataset1_name}\n"
        report_content += "=" * 80 + "\n"
        for line in self.report:
            report_content += f"{line}\n"

        st.download_button(
            label="Download Report",
            data=report_content,
            file_name=f"QAQC_report_{self.dataset1_name}.txt", 
            mime="text/markdown"
        )

    # Run all checks
    def run_checks(self):
        # Ensure the dataset is uploaded and loaded
        if self.dataset1_file:
            self.df1 = self.load_data(self.dataset1_file)
            self.dataset1_name = os.path.splitext(self.dataset1_file.name)[0]

            if self.df1 is not None:
                st.write(f"Data Loaded: {self.df1.shape[0]} rows and {self.df1.shape[1]} columns")
                mixed_data_df=self.check_mixed_data(self.df1)
                               
                st.subheader(f"{self.dataset1_name}:")
                self.check_summary(self.df1, self.dataset1_name)                            
                
                if mixed_data_df is not None:
                    st.write("🚨 Mixed Data Found:")
                    st.write(mixed_data_df)
                else:
                    st.write("✅ No mixed data found.")
                    
                    
                #st.subheader(f"Special Characters Check")
                special_chars_df=self.check_Special_Characters(self.df1)
                                             
                if special_chars_df is not None:
                    st.write("🚨 Special Characters Detected:")
                    st.write(special_chars_df)
                else:
                    st.write("✅ No Special Characters Detected.")
                                                
                st.subheader(f"Duplicates Check")
                selected_columns_dupes = st.multiselect("Select columns to check for duplicates", self.df1.columns.tolist())
                self.check_dupes(self.df1, selected_columns_dupes, self.dataset1_name) 
                
                st.subheader(f"Outliers Check")
                selected_columns_df1 = st.multiselect("Select columns to check for outliers", self.df1.columns.tolist())               
                self.check_outliers_log_iqr(self.df1, self.dataset1_name, selected_columns_df1)
                                   
                st.subheader(f"Inter File Consistency Check\n")
                inter_file_check_decision = st.radio("Would you like to perform an inter-file consistency check?", options=["Yes", "No"])
                if inter_file_check_decision == "Yes":
                    self.inter_file_check()
                                                   
                st.subheader(f"Column Name Consistency Check With Previous Year File")
                additional_file = st.file_uploader("Please upload previous year file to check column consistency", type=["csv", "xlsx"])
                if additional_file is not None:
                     self.check_column_names_match(additional_file)
                
            # After all checks, generate the report
            self.generate_report()

if __name__ == "__main__":
    tool = QAQC()
    tool.run_checks()
    
    
    
   
