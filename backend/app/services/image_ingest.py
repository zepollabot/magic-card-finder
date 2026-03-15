from typing import List

import httpx


class ImageIngestService:
    """
    Responsible for fetching listing pages and downloading images.
    This is a skeleton; implementation will be added in later tasks.
    """

    async def fetch_image_bytes_from_urls(self, urls: List[str]) -> List[bytes]:
        images: List[bytes] = []
        async with httpx.AsyncClient() as client:
            for url in urls:
                # TODO: parse HTML for image tags when given listing pages.
                resp = await client.get(url)
                resp.raise_for_status()
                images.append(resp.content)
        return images

