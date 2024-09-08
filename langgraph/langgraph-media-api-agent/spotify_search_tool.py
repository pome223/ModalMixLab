from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
import spotipy
import json
from datetime import datetime, timedelta

class SpotifySearchTool(BaseTool):
    name = "SpotifySearchTool"
    description = (
        "A tool that fetches audio features of the most recently saved tracks from Spotify. "
        "This tool does not require any arguments.\n\n"
        """Description of Return Parameters:
           acousticness: Acoustic confidence. Ex: 0.00242 (0-1)
           danceability: Dance suitability. Ex: 0.585
           duration_ms: Duration in ms. Ex: 237040
           energy: Intensity measure. Ex: 0.842
           id: Spotify track ID. Ex: "2takcwOaAZWiXQijPHIx7B"
           instrumentalness: Vocal prediction. Ex: 0.00686
           key: Track key. Ex: 9 (-1-11)
           liveness: Audience presence. Ex: 0.0866
           loudness: Loudness in dB. Ex: -5.883
           mode: Track modality. Ex: 0
           speechiness: Spoken word presence. Ex: 0.0556
           tempo: Tempo in BPM. Ex: 118.211
           time_signature: Time signature. Ex: 4 (3-7)
           type: Object type. Allowed: "audio_features"
           valence: Musical positiveness. Ex: 0.428 (0-1)
        """
    )
    args_schema: Type[BaseModel] = BaseModel  # No arguments required
    spotify_token: str = Field(..., description="Access token for Spotify")

    def __init__(self, spotify_token: str, *args, **kwargs):
        if not spotify_token:
            raise ValueError("Please set Spotify access token")
        super().__init__(spotify_token=spotify_token, *args, **kwargs)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        sp = spotipy.Spotify(auth=self.spotify_token)

        one_week_ago_date = (datetime.now() - timedelta(weeks=1)).strftime('%Y-%m-%d')
        result = sp.current_user_recently_played(limit=50, after=one_week_ago_date)

        tracks = [item['track']['id'] for item in result['items']]
        audio_features_list = [sp.audio_features(track)[0] for track in tracks]

        for i, item in enumerate(result['items']):
            track_info = item['track']
            audio_features_list[i]['song_name'] = track_info['name']
            audio_features_list[i]['artists'] = ', '.join([artist['name'] for artist in track_info['artists']])

        for features in audio_features_list:
            features.pop('uri', None)
            features.pop('track_href', None)
            features.pop('analysis_url', None)

        return json.dumps(audio_features_list)

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the SpotifySearchTool asynchronously."""
        return self._run()