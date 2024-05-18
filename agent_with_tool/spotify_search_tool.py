from langchain.tools.base import BaseTool
from pydantic import Field
from datetime import datetime, timedelta
import spotipy
import json


class SpotifySearchTool(BaseTool):
    """Tool that fetches audio features of saved tracks from Spotify."""

    name = "SpotifySearchTool"
    spotify_token: str = Field(...,
                               description="Access token for spotify.")

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

    def __init__(self, spotify_token: str, *args, **kwargs):
        if not spotify_token:
            return "Please set spotify access token"
        kwargs["spotify_token"] = spotify_token
        super().__init__(*args, **kwargs)

    def _run(self, *args, **kwargs) -> str:
        sp = spotipy.Spotify(auth=self.spotify_token)

        # 1週間前の日付を YYYY-MM-DD フォーマットで取得
        one_week_ago_date = (
            datetime.now() - timedelta(weeks=1)).strftime('%Y-%m-%d')

        result = sp.current_user_recently_played(
            limit=15, after=one_week_ago_date)

        # 仮定: result['items'] はトラックのリスト
        tracks = [item['track']['id'] for item in result['items']]

        # 各トラックのオーディオ特性を取得
        audio_features_list = [sp.audio_features(track)[0] for track in tracks]

        # 各トラックの曲名とアーティスト名を取得
        for i, item in enumerate(result['items']):
            track_info = item['track']
            song_name = track_info['name']
            artists = [artist['name'] for artist in track_info['artists']]
            audio_features_list[i]['song_name'] = song_name
            audio_features_list[i]['artists'] = ', '.join(artists)

        # uriとtrack_hrefを削除
        for features in audio_features_list:
            if 'uri' in features:
                del features['uri']
            if 'track_href' in features:
                del features['track_href']
            if 'analysis_url' in features:
                del features['analysis_url']

        # JSON形式に変換
        audio_features_json = json.dumps(audio_features_list)
        return audio_features_json

    async def _arun(self, *args, **kwargs) -> str:
        """Use the SpotifyTool asynchronously."""
        return self._run()
