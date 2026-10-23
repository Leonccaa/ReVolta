# Multimodal input limits

Media admission limits apply in the frontend. Qwen4Exp also defaults to a smaller
image processing budget, as described below. EXIF orientation is applied when
loading encoded images.
Transparent images (RGBA, LA, palette transparency and PNG tRNS) are composited
onto white before RGB conversion. White preserves the existing RGBA behavior
and matches the Qwen HF example workflow's
[`qwen_vl_utils.to_rgb`](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py).
The generic Transformers RGB converter does not itself composite alpha.
`rgba_background_color` in image media IO kwargs can select another RGB color.

## Defaults and overrides

Unspecified modality counts default to **8 images, 1 video and 1 audio item**
(except on V100 with FlashAttention-V100, see below); other inherited modalities
retain 999. `--limit-mm-per-prompt` overrides each
specified modality; a zero count disables that modality. For example,
`--limit-mm-per-prompt '{"video": 1}'` also limits images to 8 and audio to 1.
The 8-image limit counts the **whole request**, including images in historical
chat messages, rather than only the latest message.
These limits remain independent of the aggregate raw-media item ceiling below.
Embedding-only and UUID-only cache references do not fetch raw media and are
outside the raw-media byte accounting; their existing frontend limits still apply.

**V100 with FlashAttention-V100 (the ReVolta target).** On pre-Ampere GPUs with the
V100 attention path (`VLLM_SM70_FLASH_ATTN_V100=1`, the default, and the attention
backend unset, `FLASH_ATTN_V100` or `FLASHINFER_SM70`), the engine replaces the
defaults above:

- Without `--limit-mm-per-prompt`, it allows **1 image and no video** per request and
  logs `Using SM70 Flash-V100 multimodal default: image=1, video=0`. A second image is
  rejected with `At most 1 image(s) may be provided in one prompt`.
- With `--limit-mm-per-prompt` that leaves out `"video"`, video is set to 0 (disabled).
  Other modalities the value leaves out keep the defaults above.

Set `--limit-mm-per-prompt` explicitly and include `"video"`. The README launch command
uses `--limit-mm-per-prompt '{"image": 16, "video": 1, "audio": 16}'`. This behaviour is
the same as in 0.1.0 and 0.1.1. Aligning the V100 default with 8 images, 1 video and
1 audio item is planned after validation. The image resizing below applies on V100 too.

Qwen4Exp defaults to **1,048,576 pixels (1024 × 1024 in area) per image** at the
HF image processor. Larger decoded images are resized, rather than rejected by
this processing budget. HF's existing `smart_resize` preserves the aspect ratio
subject to alignment to 32-pixel multiples. The checkpoint's existing
preprocessing still applies, including alignment and minimum-size rules;
for example, 800 × 600 still becomes 800 × 608. Small text can become blurry
when a large image is reduced. Crop the relevant region or raise the processing
budget when fine detail matters. Near the area boundary, even a source image
below 1,048,576 pixels can change if HF's 32-pixel rounding would otherwise put
it over the budget. For example, 977 × 1073 changes from 992 × 1088 under the
checkpoint budget to 960 × 1056 under the new budget.

An omitted or empty `--mm-processor-kwargs '{}'` selects this default. Explicit
image size settings (`size`, `max_pixels`, `min_pixels`, or `do_resize`, including
settings inside `images_kwargs`) retain operator precedence. To restore the
checkpoint's 16,777,216-pixel image budget, use:

```bash
--mm-processor-kwargs '{"images_kwargs": {"size": {"longest_edge": 16777216, "shortest_edge": 65536}}}'
```

The image default leaves the video processor's size unchanged. Generic top-level
size overrides can also affect video processing through HF's existing routing;
use `images_kwargs` for an image-only override. Startup logs report the effective
image pixel budget and its source. The image dummy profile uses this same budget
(1024 maximum image tokens by default instead of 16384 for the checkpoint).
Encoder and KV memory capacity still depend on enabled video inputs and measured
GPU allocations; a smaller image budget does not imply a fixed KV increase.

| Server environment variable | Default | Scope |
| --- | ---: | --- |
| `VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB` | 256 MiB | One remote response, including decompressed HTTP content and cached response bytes |
| `VLLM_MAX_MEDIA_REQUEST_BYTES` | 268435456 B (256 MiB) | Aggregate encoded bytes from HTTP, data URLs, files and download-cache hits |
| `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES` | 536870912 B (512 MiB) | Aggregate image and sampled-video raster accounting |
| `VLLM_MAX_MEDIA_REQUEST_ITEMS` | 32 | Aggregate fetched raw-media items across modalities |
| `VLLM_MAX_IMAGE_PIXELS` | 16777216 | Pixels per image or video frame |
| `VLLM_MAX_VIDEO_FRAMES` | 64 | Sampled frames per video |

These environment values are operator settings. Set zero to disable a particular
ceiling explicitly. Request `media_io_kwargs` cannot raise them. Count overrides
also cannot bypass byte, pixel or frame ceilings. The single-response limit
prevents one large download; the request limit prevents many smaller inputs.
Repeated URLs and cache hits are charged again. Failed attempts conservatively
retain their charges for the current request; a new request has a fresh budget.
HTTP size errors are non-retryable validation failures and map to HTTP 400.

Non-empty per-request `mm_processor_kwargs` and `media_io_kwargs` are rejected
with HTTP 400 by default. Only deployments with trusted API clients should enable
`--trust-request-mm-kwargs` to allow these overrides. Omitted values, `null` and
empty dictionaries retain their previous behavior. Server-level
`--mm-processor-kwargs` and `--media-io-kwargs` remain operator settings.
The same policy applies to render, tokenize, pooling and `run-batch` frontends;
the local Python `LLM` API keeps accepting caller-controlled overrides.

`--media-io-kwargs '{"video": {"num_frames": 16}}'` remains a default sampling
setting. With `--trust-request-mm-kwargs`, requests can override `num_frames` or
`fps` using the existing merge rules: overriding either clears the other
inherited field. The loader's fallback
is 16 frames when `num_frames` is omitted. The operator's 64-frame ceiling
and raster budget apply after these merges. `num_frames=-1` can load all frames only when the actual selected count fits
the frame ceiling and raster budget; this is checked before frame allocation.
Backends without pre-decode accounting are rejected while the decoded budget is
enabled. This does not add a support promise for inherited decoder backends.

## Allocation boundaries

Remote bodies are streamed in 64 KiB chunks. Content-Length is checked before
reading; missing lengths and gzip/deflate responses are counted incrementally.
Decompression steps have bounded output. Files and cached responses are also
read incrementally; data URL size is checked before base64 allocation.

Images are charged at four bytes per pixel before EXIF processing or raster
loading. The common OpenCV/PyAV video loader checks the selected frame count and
frame dimensions before allocating output frame arrays. JPEG frame sequences
charge each image before loading and check frame count before stacking.
The default decoded budget accommodates 16 frames of 3840 x 2160 at four bytes
per pixel (530841600 B), or several common 12-megapixel images. It does not
promise that a long source video fits the separate 256 MiB download ceiling.

These are admission and retained-data limits, not a hard process RSS ceiling.
Video codecs can allocate internal reference frames and scratch buffers; inaccurate
container dimensions can only be detected after decoding. Image conversion,
frame stacking and joining encoded chunks also create temporary copies.
Concurrent requests each have their own budget. Operators must additionally
bound HTTP request bodies, concurrent requests and process/container memory.
Audio encoded bytes are counted; inherited audio decode allocation is outside
this image/video change. Cache identity and rejection recovery are unchanged.
