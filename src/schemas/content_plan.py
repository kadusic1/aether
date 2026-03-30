from pydantic import BaseModel, Field
from typing import Literal


class Scene(BaseModel):
    """
    A single scene in a short-form video.
    """

    narration: str = Field(
        description=(
            "Narration text spoken aloud during this"
            "MUST be extremely fast-paced. "
            "Zero idle moments or fluff. Use punchy, conversational tone. "
            "Include mini-hooks or sudden reveals to maintain retention."
        ),
    )
    image_prompt: str = Field(
        description=(
            "DETAILED prompt for the image generation model."
            " Vivid, specific, cinematic. Include"
            " style keywords (dramatic lighting,"
            " close-up, moody atmosphere, etc)."
            " NEVER include text, words, or letters"
            " in the image. Describe only visual"
            " elements."
        ),
    )
    image_motion: Literal[
        "zoom_in",
        "zoom_out",
        "pan_left",
        "pan_right",
        "static",
    ] = Field(
        description=(
            "Ken Burns camera motion applied to the"
            " still image. zoom_in for emphasis,"
            " zoom_out for reveals, pan for"
            " landscapes or movement. Use static for"
            " high-detail images where motion would distract."
        ),
    )


class ContentPlan(BaseModel):
    """
    Complete production plan for a short-form video.
    """

    video_name: str = Field(
        description=(
            "Short, catchy, curiosity-driven title"
            " optimized for viral Shorts/Reels/TikToks discovery."
        ),
    )
    thumbnail_prompt: str = Field(
        description=(
            "DETAILED image generation prompt for the video"
            " thumbnail. Eye-catching, bold, high"
            " contrast. Close-up face or dramatic"
            " visual. NEVER include text or words."
            " MUST visually match the FIRST scene"
        ),
    )
    scenes: list[Scene] = Field(
        description=(
            "Ordered list of scenes forming a"
            " continuous narrative. The FIRST scene"
            " MUST be a 3-second scroll-stopping"
            " hook. Scenes flow naturally into each"
            " other."
        ),
        min_length=8,
    )

    def __str__(self):
        content_string = "Video: " + self.video_name + "\n"
        content_string += "Thumbnail: " + self.thumbnail_prompt + "\n"
        content_string += "Number of scenes: " + str(len(self.scenes)) + "\n\n"
        for i, scene in enumerate(self.scenes):
            content_string += f"{i + 1}: " + scene.narration + "\n"
            content_string += (
                f"[img] [{scene.image_motion}] " + scene.image_prompt + "\n\n"
            )
        return content_string
