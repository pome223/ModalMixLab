from langchain.tools.base import BaseTool
from google.cloud import bigquery
from google.oauth2 import service_account
from pydantic import Field
import datetime
import json


class BigQueryWriteTool(BaseTool):
    """Tool that writes data to Google BigQuery."""

    name: str = "BigQueryWriteTool"
    bigquery_credentials_file: str = Field(...,
                                           description="Path to BigQuery credentials file.")
    dataset_name: str = Field(..., description="BigQuery dataset name.")
    table_name: str = Field(..., description="BigQuery table name.")
    description: str = (
        # "A tool that writes data to Google BigQuery.\n"
        "In English, that would be:This tool summarizes the conversation between the user and the AI assistant and registers it in BigQuery."
        # "The BigQueryWriteTool takes a dictionary with two keys, 'topics' and 'keywords', and uses it to process the data."
        "Arguments:\n"
        "smmry_cnvn: This is a summary of the conversation between the user and the assistant (character limit is 100 characters)."
        #   "- data: A dictionary with two keys, 'topics' and 'keywords'. "
        #   "Each key should have a list of strings as its value.\n\n"
        #   """
        #   Example:
        #       data = {
        #           'topics': ['topic1', 'topic2'],
        #           'keywords': ['keyword1', 'keyword2']
        #       }
        #   """
        "Output:\n"
        "insert job return result status."
    )

    def __init__(self, bigquery_credentials_file: str, dataset_name: str, table_name: str, *args, **kwargs):
        if not bigquery_credentials_file or not dataset_name or not table_name:
            raise ValueError(
                "BigQuery credential, dataset and table must be provided.")

        kwargs["bigquery_credentials_file"] = bigquery_credentials_file
        kwargs["dataset_name"] = dataset_name
        kwargs["table_name"] = table_name

        super().__init__(*args, **kwargs)

    def _run(self, smmry_cnvn: str):
        if len(smmry_cnvn) > 100:
            return "The summary conversation is over the character limit."

        # try:
            # JSON convert str to dict
            # data = json.loads(data)
        # except json.JSONDecodeError:
            # raise ValueError("Data is not a valid JSON string")

        # if not all(key in data for key in ['topics', 'keywords']):
            # raise ValueError("Data must contain 'topics' and 'keywords' keys")

        # Write the data to BigQuery
        credentials = service_account.Credentials.from_service_account_file(
            self.bigquery_credentials_file)
        client = bigquery.Client(
            credentials=credentials, project=credentials.project_id)

        table_ref = client.dataset(self.dataset_name).table(self.table_name)
        table = client.get_table(table_ref)

        # create data
        rows_to_insert = [
            #   (datetime.datetime.now(), data['topics'], data['keywords']),
            (datetime.datetime.now(), smmry_cnvn),
        ]

        # Insert the data into the table
        errors = client.insert_rows(table, rows_to_insert)

        message = 'New rows have been added.'

        # Check for errors
        if errors != []:
            message = 'Encountered errors while inserting rows: {}'.format(
                errors)

        return message

    async def _arun(self, rows_to_insert) -> str:
        """Use the BigQueryWriteTool asynchronously."""
        return self._run(rows_to_insert)
