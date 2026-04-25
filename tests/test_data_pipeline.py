import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from sentinelhub import CRS, BBox

from data_pipeline.image_utils import (
    compute_split_bboxes,
    extract_bands_from_response,
    save_geotiff,
    stitch_tiles,
    to_rgb,
)
from data_pipeline.sentinel_request import fetch_token, make_json, make_request
from data_pipeline.sentinel_utils import compute_nbr

test_limits= [np.float64(100.92801), np.float64(17.90662), np.float64(102.77297999999999), np.float64(19.49714)]

def _make_tiff_bytes(array: np.ndarray) -> bytes:
    height, width = array.shape[1:3]
    count = array.shape[0]

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=array.dtype,
        ) as dataset:
            for index in range(count):
                dataset.write(array[index], index + 1)
        return memfile.read()


class TestSentinelUtils(unittest.TestCase):
    def test_compute_nbr_masks_invalid_pixels_and_handles_zero_division(self):
        bands = np.zeros((7, 2, 2), dtype=np.float32)
        bands[3] = np.array([[8.0, 0.0], [4.0, 2.0]], dtype=np.float32)
        bands[5] = np.array([[2.0, 0.0], [4.0, 2.0]], dtype=np.float32)
        bands[6] = np.array([[1, 1], [0, 1]], dtype=np.uint8)

        nbr = compute_nbr(bands)

        self.assertAlmostEqual(nbr[0, 0], 0.6)
        self.assertTrue(np.isnan(nbr[0, 1]))
        self.assertTrue(np.isnan(nbr[1, 0]))
        self.assertAlmostEqual(nbr[1, 1], 0.0)


class TestImageUtils(unittest.TestCase):
    def test_to_rgb_reorders_and_clips_channels(self):
        bands = np.zeros((7, 1, 2), dtype=np.float32)
        bands[0, 0, 0] = 0.4
        bands[1, 0, 0] = 0.2
        bands[2, 0, 0] = 0.1
        bands[0, 0, 1] = 0.5
        bands[1, 0, 1] = 0.5
        bands[2, 0, 1] = 0.5

        rgb = to_rgb(bands)

        self.assertEqual(rgb.shape, (3, 1, 2))
        self.assertTrue(np.allclose(rgb[:, 0, 0], np.array([0.35, 0.7, 1.0])))
        self.assertTrue(np.allclose(rgb[:, 0, 1], np.array([1.0, 1.0, 1.0])))

    def test_extract_bands_from_response_reads_tiff_content(self):
        bands = np.arange(12, dtype=np.uint16).reshape(3, 2, 2)
        response = SimpleNamespace(
            status_code=200,
            content=_make_tiff_bytes(bands),
        )

        extracted = extract_bands_from_response(response)

        self.assertTrue(np.array_equal(extracted, bands))

    def test_compute_split_bboxes_builds_expected_grid(self):
        bbox = BBox([0.0, 0.0, 10.0, 10.0], crs=CRS.WGS84)

        with patch(
            "data_pipeline.image_utils.bbox_to_dimensions",
            return_value=(5000, 4000),
        ):
            sub_bboxes, n_rows, n_cols = compute_split_bboxes(
                bbox,
                resolution=10,
                max_pixels=2500,
            )

        self.assertEqual((n_rows, n_cols), (2, 2))
        self.assertEqual(len(sub_bboxes), 4)
        self.assertAlmostEqual(sub_bboxes[0].min_x, 0.0)
        self.assertAlmostEqual(sub_bboxes[0].min_y, 0.0)
        self.assertAlmostEqual(sub_bboxes[0].max_x, 5.0)
        self.assertAlmostEqual(sub_bboxes[0].max_y, 5.0)
        self.assertAlmostEqual(sub_bboxes[-1].max_x, 10.0)
        self.assertAlmostEqual(sub_bboxes[-1].max_y, 10.0)

    def test_stitch_tiles_concatenates_row_major_tiles(self):
        tile_1 = np.full((2, 2, 1), 1, dtype=np.uint8)
        tile_2 = np.full((2, 1, 1), 2, dtype=np.uint8)
        tile_3 = np.full((1, 2, 1), 3, dtype=np.uint8)
        tile_4 = np.full((1, 1, 1), 4, dtype=np.uint8)

        stitched = stitch_tiles(
            [tile_1, tile_2, tile_3, tile_4],
            n_rows=2,
            n_cols=2,
        )

        expected = np.array(
            [
                [[1], [1], [2]],
                [[1], [1], [2]],
                [[3], [3], [4]],
            ],
            dtype=np.uint8,
        )
        self.assertTrue(np.array_equal(stitched, expected))

    def test_save_geotiff_round_trips_data(self):
        bbox = BBox([1.0, 2.0, 3.0, 4.0], crs=CRS.WGS84)
        data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)

        with tempfile.TemporaryDirectory() as tempdir:
            filename = f"{tempdir}/test.tif"
            save_geotiff(data, filename, bbox)

            with rasterio.open(filename) as src:
                self.assertEqual(src.count, 2)
                self.assertEqual((src.width, src.height), (2, 2))
                round_trip = src.read()

        self.assertTrue(np.array_equal(round_trip, data))


class TestSentinelRequest(unittest.TestCase):
    def test_make_json_uses_bbox_bounds_and_date_window(self):
        bbox = BBox(test_limits, crs=CRS.WGS84)

        with patch(
            "data_pipeline.sentinel_request.bbox_to_dimensions",
            return_value=(60, 50),
        ):
            request = make_json(
                bbox,
                "2023-06-01",
                "2023-07-01",
                50,
                60,
            )

        self.assertEqual(
            request["input"]["bounds"]["bbox"],
            test_limits,
        )
        self.assertEqual(
            request["input"]["data"][0]["dataFilter"]["timeRange"],
            {
                "from": "2023-06-01T00:00:00Z",
                "to": "2023-07-01T23:59:59Z",
            },
        )
        self.assertEqual(request["input"]["data"][0]["type"], "sentinel-2-l2a")
        self.assertEqual(request["output"]["width"], 60)
        self.assertEqual(request["output"]["height"], 50)
        self.assertIsInstance(request["evalscript"], str)


class TestSentinelFetchToken(unittest.TestCase):
    @patch("data_pipeline.sentinel_request.OAuth2Session")
    @patch("data_pipeline.sentinel_request.BackendApplicationClient")
    @patch("data_pipeline.sentinel_request.os.getenv")
    def test_fetch_token_returns_valid_token(
        self,
        mock_getenv,
        mock_backend_client,
        mock_oauth_session,
    ):
        values = {
            "sentinel_client_id": "client-id",
            "sentinel_client_secret": "client-secret",
            "sentinel_token_url": "https://example.com/token",
        }
        mock_getenv.side_effect = lambda key: values.get(key)

        oauth_instance = Mock()
        oauth_instance.fetch_token.return_value = {
            "access_token": "fake-access-token",
            "expires_at": 1_900_000_000,
        }
        mock_oauth_session.return_value = oauth_instance

        token = fetch_token()

        self.assertIsNotNone(token)
        self.assertIn("access_token", token)
        self.assertEqual(os.environ.get("sentinel_access_token"), "fake-access-token")
        self.assertEqual(os.environ.get("sentinel_token_expiry"), "1900000000")
        mock_backend_client.assert_called_once_with(client_id="client-id")
        oauth_instance.fetch_token.assert_called_once_with(
            token_url="https://example.com/token",
            client_secret="client-secret",
            include_client_id=True,
        )

class TestSentinelMakeRequest(unittest.TestCase):
    @patch("data_pipeline.sentinel_request.os.getenv", return_value=None)
    def test_make_request_without_cached_token_raises_exception(self, _mock_getenv):
        with patch(
            "data_pipeline.sentinel_request.fetch_token",
            return_value=None,
        ):
            with self.assertRaises(Exception) as context:
                make_request({})
            self.assertIn(
                "Failed to fetch access token for Sentinel API",
                str(context.exception),
            )

    @patch("data_pipeline.sentinel_request.requests.post")
    @patch("data_pipeline.sentinel_request.fetch_token")
    @patch("data_pipeline.sentinel_request.os.getenv")
    def test_make_request_uses_refreshed_token_and_posts(
        self,
        mock_getenv,
        mock_fetch_token,
        mock_post,
    ):
        def getenv_side_effect(key, default=None):
            mapping = {
                "sentinel_access_token": "stale-token",
                "sentinel_token_expiry": "0",
                "sentinel_request_url": "https://example.com/process",
            }
            return mapping.get(key, default)

        mock_getenv.side_effect = getenv_side_effect
        mock_fetch_token.return_value = {
            "access_token": "fresh-token",
            "expires_at": 1_900_000_000,
        }

        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        result = make_request({"hello": "world"})

        self.assertEqual(result.status_code, 200)
        mock_fetch_token.assert_called_once()
        mock_post.assert_called_once_with(
            url="https://example.com/process",
            headers={
                "Authorization": "Bearer fresh-token",
                "Content-Type": "application/json",
            },
            json={"hello": "world"},
        )

if __name__ == "__main__":
    unittest.main()
