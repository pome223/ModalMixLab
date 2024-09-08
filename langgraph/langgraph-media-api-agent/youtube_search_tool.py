from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import json

class YouTubeSearchInput(BaseModel):
    query: str = Field(description="The search term to look for on YouTube")
    max_results: int = Field(default=100, description="Maximum number of results to fetch")

class YouTubeSearchTool(BaseTool):
    name = "YoutubeSearchTool"
    description = (
        "A tool that fetches search results from YouTube based on a query.\n"
        "Output Format:\n"
        "- Title: Displayed after translation to Japanese.\n"
        "- first_280_chars_of_transcript: This field contains the first 280 characters of the video's transcript.\n"
        "- viewCount: Number of times the video has been viewed.\n"
        "- likeCount: Number of likes the video has received.\n"
        "- Description: Displayed after translation to Japanese.\n"
        "- Published Date: Displayed as 'publishedAt'.\n"
        "- Video Link: Formatted as https://www.youtube.com/watch?v={video_id}."
    )
    args_schema: Type[BaseModel] = YouTubeSearchInput
    youtube_api_key: str = Field(..., description="API key for accessing Youtube data.")

    def __init__(self, youtube_api_key: str, *args, **kwargs):
        if not youtube_api_key:
            raise ValueError("A valid Youtube developer key must be provided.")
        super().__init__(youtube_api_key=youtube_api_key, *args, **kwargs)

    def _run(
        self,
        query: str,
        max_results: int = 100,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        YOUTUBE_API_SERVICE_NAME = "youtube"
        YOUTUBE_API_VERSION = "v3"
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=self.youtube_api_key)

        search_response = youtube.search().list(
            q=query,
            part="id,snippet",
            order='date',
            type='video',
            maxResults=max_results
        ).execute()

        videos = search_response['items']
        video_list = []

        for video in videos:
            video_data = {}
            video_id = video['id']['videoId']
            video_data['video_id'] = video_id
            video_data['title'] = video['snippet']['title']
            video_data['publishedAt'] = video['snippet']['publishedAt']
            video_data['description'] = video['snippet']['description']

            video_response = youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()
            statistics = video_response["items"][0]["statistics"]
            video_data['viewCount'] = statistics.get("viewCount", "0")
            video_data['likeCount'] = statistics.get("likeCount", "0")

            if int(video_data['viewCount']) >= 1000:
                video_list.append(video_data)

        latest_5_videos = sorted(video_list, key=lambda x: x['publishedAt'], reverse=True)[:5]

        for video in latest_5_videos:
            video_id = video['video_id']
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'ja'])
                transcript_text = [entry['text'] for entry in transcript]
                transcript_string = ' '.join(transcript_text)
                first_280_chars = transcript_string[:280]
                video['first_280_chars_of_transcript'] = first_280_chars
            except:
                video['first_280_chars_of_transcript'] = "Transcript not available"

        return json.dumps(latest_5_videos)

    async def _arun(
        self,
        query: str,
        max_results: int = 100,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(query, max_results)