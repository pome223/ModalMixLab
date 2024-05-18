from langchain.tools.base import BaseTool
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import json
from pydantic import Field


class YoutubeSearchTool(BaseTool):
    """Tool that fetches search results from YouTube."""

    name: str = "YoutubeSearchTool"
    youtube_api_key: str = Field(...,
                                 description="API key for accessing Youtube data.")
    description: str = (
        "A tool that fetches search results from YouTube based on a query.\n"
        "Arguments:\n"
        "- query: The search term to look for on YouTube.\n"
        "- youtube_api_key: The API key to access YouTube data.\n\n"
        "Output Format:\n"
        "- Title: Displayed after translation to Japanese.\n"
        "- first_280_chars_of_transcript:This field contains the first 280 characters of the video's transcript.\n"
        "- viewCount: Number of times the video has been viewed.\n"
        "- likeCount: Number of likes the video has received.\n"
        "- Description: Displayed after translation to Japanese.\n"
        "- Published Date: Displayed as 'publishedAt'.\n"
        "- Video Link: Formatted as https://www.youtube.com/watch?v={video_id}."
    )

    def __init__(self, youtube_api_key: str, *args, **kwargs):
        if not youtube_api_key:
            raise ValueError("A valid Youtube developer key must be provided.")
        kwargs["youtube_api_key"] = youtube_api_key
        super().__init__(*args, **kwargs)

    def _run(self, q: str, max_results: int = 100) -> str:
        YOUTUBE_API_SERVICE_NAME = "youtube"
        YOUTUBE_API_VERSION = "v3"
        youtube = build(YOUTUBE_API_SERVICE_NAME,
                        YOUTUBE_API_VERSION, developerKey=self.youtube_api_key)

        search_response = youtube.search().list(
            q=q,
            part="id,snippet",
            order='date',  # Sort by published date
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

            # Fetch viewCount and likeCount for each video
            video_response = youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()
            statistics = video_response["items"][0]["statistics"]
            video_data['viewCount'] = statistics.get("viewCount", "0")
            video_data['likeCount'] = statistics.get("likeCount", "0")

            # Only add videos with more than 1000 views to the list
            if int(video_data['viewCount']) >= 1000:
                video_list.append(video_data)

        # Sort the video list by 'publishedAt' in descending order and take the first 5
        latest_5_videos = sorted(
            video_list, key=lambda x: x['publishedAt'], reverse=True)[:5]

        # Get first 280 characters of transcript for each video
        for video in latest_5_videos:
            video_id = video['video_id']
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=['en', 'ja'])
                transcript_text = [entry['text'] for entry in transcript]
                transcript_string = ' '.join(transcript_text)
                first_280_chars = transcript_string[:280]
                video['first_280_chars_of_transcript'] = first_280_chars
            except:
                video['first_280_chars_of_transcript'] = "Transcript not available"

        # Convert to JSON format
        items_json = json.dumps(latest_5_videos)
        return items_json

    async def _arun(self, q: str, max_results: int = 100) -> str:
        """Use the YoutubeSearchTool asynchronously."""
        return self._run(q, max_results)
