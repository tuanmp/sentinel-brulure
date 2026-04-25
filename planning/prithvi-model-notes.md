# Prithvi-EO-1.0-100M-burn-scar Model Research

## Source

- HuggingFace: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-burn-scar
- Pre-trained by IBM + NASA on HLS (Harmonized Landsat Sentinel-2) data

## Can It Be Used Without Fine-Tuning?

**Yes.** The model is already fine-tuned on the HLS Burn Scar dataset and achieves:
- **Burn scar IoU**: 0.73
- **Overall accuracy**: 0.96

## Input Requirements

| Field | Value |
|-------|-------|
| Shape | 512×512×6 (H×W×C) |
| Bands (in order) | Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2 |
| Values | Reflectance units [0-1] |
| Format | GeoTIFF |

## Our Data Match — Band Mapping

| Sentinel-2 Band | Index | Prithvi Model Input | Status |
|-----------------|-------|------------------|--------|
| B02 | 0 | Blue | ✅ |
| B03 | 1 | Green | ✅ |
| B04 | 2 | Red | ✅ |
| B08 | 3 | Narrow NIR | ✅ |
| B11 | 4 | SWIR 1 | ✅ |
| B12 | 5 | SWIR 2 | ✅ |
| dataMask | 6 | N/A (metadata) | ❌ Drop |

**Note:** Our output has 7 bands. Band 6 is `dataMask` (binary: 0=invalid, 1=valid) — this is metadata, not a spectral band. We already use it to mask invalid pixels in NBR computation.

## What To Change in Our Pipeline

1. **Drop dataMask** — take only bands 0-5 from our 7-band output
2. **Resize to 512×512** — Prithvi expects exactly 512×512 input
3. **Keep values as-is** — model expects [0-1] reflectance (which is what we get)

## Alternative: Use dNBR Directly

Since our pipeline already computes `dNBR` (differential Normalized Burn Ratio), we can use it as a **weak supervision signal** without the model:
- dNBR > 0.1: low burn severity
- dNBR > 0.27: moderate burn severity
- dNBR > 0.44: high burn severity
- dNBR > 0.66: severe burn

This doesn't require ML inference at all — just threshold the dNBR raster.

## Model Inference Dependencies

To run inference, need to install:
```
mmcv-full
mmsegmentation
torch
rasterio
```

## Inference Script

Available in model GitHub repo. Basic usage:
```python
python burn_scar_batch_inference_script.py \
  -config burn_scars_Prithvi_100M.py \
  -ckpt burn_scars_Prithvi_100M.pth \
  -input /path/to/512x512x6/geotiff \
  -output /path/to/output
```

## Next Steps

1. If using dNBR only: skip ML model entirely — threshold dNBR raster
2. If using model: modify pipeline to output 6 bands at 512×512
3. Run inference on post-fire tensor

## References

- Model: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-burn-scar
- GitHub: https://github.com/NASA-IMPACT/hls-foundation-os
- HLS dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars