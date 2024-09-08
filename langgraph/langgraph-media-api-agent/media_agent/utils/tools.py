import os
from langchain_community.tools.tavily_search import TavilySearchResults
from youtube_search_tool import YouTubeSearchTool
from spotify_search_tool import SpotifySearchTool
from spotify_playlist_tool import SpotifyPlaylistTool
# tools = [TavilySearchResults(max_results=1),YouTubeSearchTool(youtube_api_key = os.getenv('YOUTUBE_API'))]
tools = [TavilySearchResults(max_results=1),YouTubeSearchTool(youtube_api_key = os.getenv('YOUTUBE_API')), SpotifySearchTool(spotify_token= os.getenv('SPOTIFY_TOKEN')),SpotifyPlaylistTool(user_id =  os.getenv('SPOTIFY_CLIENTID'),spotify_token= os.getenv('SPOTIFY_TOKEN'))]