from langchain.tools.base import BaseTool
from pydantic import Field
import tweepy


class TwitterPostTool(BaseTool):
    """Tool that posts a tweet on X (formerly Twitter)."""

    name: str = "TwitterPostTool"
    consumer_key: str = Field(...,
                              description="Consumer Key for accessing X API.")
    consumer_secret: str = Field(...,
                                 description="Consumer Secret for accessing X API.")
    access_token: str = Field(...,
                              description="Access Token for accessing X API.")
    access_token_secret: str = Field(...,
                                     description="Access Token Secret for accessing X API.")
    description: str = (
        "Before using this tool to tweet, first ask the user to review the content of the 'text' argument.\n\n"
        "A tool that posts a tweet on X.\n"
        "Arguments:\n"
        "- text: The text of the tweet. (Must be must be 280 characters or less for 1-byte characters, and 140 characters or less for 2-byte characters)\n\n"
        "Output Format:\n"
        "- Tweet URL: The URL of the posted tweet, formatted as tweet_url."
    )

    def __init__(self, consumer_key: str, consumer_secret: str, access_token: str, access_token_secret: str, *args, **kwargs):
        if not consumer_key or not consumer_secret or not access_token or not access_token_secret:
            raise ValueError("All X API keys and tokens must be provided.")
        kwargs["consumer_key"] = consumer_key
        kwargs["consumer_secret"] = consumer_secret
        kwargs["access_token"] = access_token
        kwargs["access_token_secret"] = access_token_secret
        super().__init__(*args, **kwargs)

    def _run(self, text: str) -> str:
        text_length = sum(2 if ord(c) > 0x7f else 1 for c in text)
        if text_length >= 280:
            return "The text argument must be 280 characters or less for 1-byte characters, and 140 characters or less for 2-byte characters"

        client = tweepy.Client(
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
        )

        # Post the tweet
        response = client.create_tweet(text=text)
        tweet_id = response.data['id']

        # Get user_id
        username = client.get_me().data.username

        tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"
        return tweet_url
