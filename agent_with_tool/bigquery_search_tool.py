from google.cloud import bigquery
from google.oauth2 import service_account
from pydantic import Field
from langchain.tools.base import BaseTool


class BigQuerySearchTool(BaseTool):
    """ 
    Tool for searching data in Google BigQuery.

    This tool is designed to perform search operations on the 'smmry_cnvn' column in a BigQuery table.
    The primary purpose is to help users quickly find relevant entries based on their search terms.

    Attributes:
    - bigquery_credentials_file (str): Path to the BigQuery credentials file.
    - dataset_name (str): Name of the BigQuery dataset.
    - table_name (str): Name of the BigQuery table within the dataset.
    - description (str): Describes the function of the tool and its parameters.
    """

    bigquery_credentials_file: str = Field(...,
                                           description="Path to BigQuery credentials file.")
    dataset_name: str = Field(..., description="BigQuery dataset name.")
    table_name: str = Field(..., description="BigQuery table name.")
    description: str = """
        This tool allows you to search in the 'smmry_cnvn' and 'timestamp' columns of a BigQuery table.
        The 'search_term' should be provided as a condition for the WHERE clause. For instance:
        
        - smmry_cnvn represents a summary conversation.
        - timestamp indicates the created time.
        
        Example:
            SELECT * 
            FROM `dataset.table` 
            WHERE <search_term>
        
        In this example, `<search_term>` could be "smmry_cnvn LIKE '%some_keyword%'" or "timestamp > '2023-01-01'". 
    """

    def _run(self, search_term: str):
        """Search for entries in the BigQuery table using the provided search term."""
        # Initialize BigQuery client
        credentials = service_account.Credentials.from_service_account_file(
            self.bigquery_credentials_file)
        client = bigquery.Client(
            credentials=credentials, project=credentials.project_id)

        # Create the search query
        # query =''

        query = f"""
            SELECT * 
            FROM `{credentials.project_id}.{self.dataset_name}.{self.table_name}` 
            WHERE {search_term}
        """

        # if query_type == 'keyword':
        #     query = f"""
        #         SELECT *
        #         FROM `{credentials.project_id}.{self.dataset_name}.{self.table_name}`
        #         WHERE smmry_cnvn LIKE @search_term
        #     """
        # else query_type == "timestamp";
        #      query = f"""
        #         SELECT *
        #         FROM `{credentials.project_id}.{self.dataset_name}.{self.table_name}`
        #         WHERE created_at < @search_term
        #     """

        # Use parameterized query to avoid SQL injection
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "search_term", "STRING", f"{search_term}")
            ]
        )

        # Execute the query
        # query_job = client.query(query, job_config=job_config)
        query_job = client.query(query)

        results = query_job.result()

        # Return results as a list
        return [row.smmry_cnvn for row in results]

    async def _arun(self, search_term: str) -> list:
        """Use the BigQuerySearchTool asynchronously."""
        return self._run(search_term)

# Usage example:
# tool = BigQuerySearchTool(bigquery_credentials_file="path_to_your_service_account_key.json", dataset_name="your_dataset_name", table_name="your_table_name")
# search_results = await tool._arun("desired_search_term")
# print(search_results)
