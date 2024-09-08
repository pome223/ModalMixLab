from typing import Optional, Type, List
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
import spotipy

class SpotifyPlaylistInput(BaseModel):
    track_ids: List[str] = Field(description="List of Spotify track IDs to add to the playlist")
    playlist_name: str = Field(description="Name of the new playlist to be created")
    playlist_description: str = Field(description="Description for the new playlist")

class SpotifyPlaylistTool(BaseTool):
    name = "SpotifyPlaylistTool"
    description = (
        "A tool that creates a new playlist and adds tracks to it on Spotify. "
        "This tool requires a list of track IDs, a playlist name, and a playlist description."
    )
    args_schema: Type[BaseModel] = SpotifyPlaylistInput
    spotify_token: str = Field(..., description="Access token for Spotify")
    user_id: str = Field(..., description="User ID for Spotify")

    def __init__(self, spotify_token: str, user_id: str, *args, **kwargs):
        if not spotify_token:
            raise ValueError("Please set Spotify access token")
        if not user_id:
            raise ValueError("Please set Spotify user ID")
        super().__init__(spotify_token=spotify_token, user_id=user_id, *args, **kwargs)

    def _run(
        self,
        track_ids: List[str],
        playlist_name: str,
        playlist_description: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        sp = spotipy.Spotify(auth=self.spotify_token)

        # Create a new playlist
        user_playlist = sp.user_playlist_create(self.user_id, playlist_name, public=False, collaborative=False, description=playlist_description)

        # Add tracks to the playlist
        sp.playlist_add_items(user_playlist['id'], items=track_ids, position=None)

        return f"Playlist '{playlist_name}' created with {len(track_ids)} tracks."

    async def _arun(
        self,
        track_ids: List[str],
        playlist_name: str,
        playlist_description: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the SpotifyPlaylistTool asynchronously."""
        return self._run(track_ids, playlist_name, playlist_description)