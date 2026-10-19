# Multimodal input limits

Media limits apply in the frontend. They do not change model computation for
accepted inputs. EXIF orientation is applied when loading encoded images.
Transparent images (RGBA, LA, palette transparency and PNG tRNS) are composited
onto white before RGB conversion. White preserves the existing RGBA behavior
and matches the Qwen HF example workflow's
[`qwen_vl_utils.to_rgb`](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py).
The generic Transformers RGB converter does not itself composite alpha.
`rgba_background_color` in image media IO kwargs can select another RGB color.

## Defaults and overrides

Unspecified modality counts default to **16 images, 1 video and 16 audio items**;
other inherited modalities retain 999. `--limit-mm-per-prompt` overrides each
specified modality; a zero count disables that modality. For example,
`--limit-mm-per-prompt '{"video": 1}'` now also limits images to 16.
These limits remain independent of the aggregate raw-media item ceiling below.
Embedding-only and UUID-only cache references do not fetch raw media and are
outside the raw-media byte accounting; their existing frontend limits still apply.

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

`--media-io-kwargs '{"video": {"num_frames": 16}}'` remains a default sampling
setting. Requests can override `num_frames` or `fps` using the existing merge
rules: overriding either clears the other inherited field. The loader's fallback
is still 32 frames when `num_frames` is omitted. The operator's 64-frame ceiling
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
