# Multimodal input limits

This page lists the limits that apply to image and video input and how an operator changes them: how many items a request may contain, how large images are resized, how many video frames are sampled, and the per-request byte and pixel budgets. It applies to Qwen3.8 Flash-Next, which takes images and video but not audio.

## Item counts per request

On V100 the engine allows only **1 image and no video** per request unless you set `--limit-mm-per-prompt`. This V100 default applies on pre-Ampere GPUs when the V100 attention path is active (`VLLM_SM70_FLASH_ATTN_V100=1`, the default) and the attention backend is unset, `FLASH_ATTN_V100` or `FLASHINFER_SM70`. The engine then logs `Using SM70 Flash-V100 multimodal default: image=1, video=0`, and a second image is rejected with `At most 1 image(s) may be provided in one prompt`.

So always set `--limit-mm-per-prompt` on V100, and include `"video"`: a value that leaves out `"video"` sets video to 0 (disabled).

- The [reference launch](../docs/getting-started.md#4-launch) uses `--limit-mm-per-prompt '{"image": 16, "video": 1}'`.
- `--limit-mm-per-prompt '{"image": 8, "video": 1}'` gives the generic vLLM counts. A lower image count leaves more host memory for bursts of large images.

Other rules:

- The count covers the **whole request**, including images in earlier chat messages, not only the latest message.
- A count of zero disables that modality.
- Modalities that the option leaves out default to 8 images and 1 audio item, and to 1 video except on V100 (where video becomes 0); other modalities default to 999. For example, `'{"video": 1}'` allows 8 images.
- These counts are separate from the request-wide item ceiling `VLLM_MAX_MEDIA_REQUEST_ITEMS` (below). Count overrides cannot bypass the byte, pixel or frame ceilings.
- Embedding-only and UUID-only cache references do not fetch raw media and are outside the raw-media byte accounting; the count limits still apply to them.

## Image size

The model's image processor resizes each image to at most **1,048,576 pixels (1024 × 1024 in area)**. Larger decoded images are resized, not rejected. HF's `smart_resize` keeps the aspect ratio and aligns both sides to multiples of 32 pixels, and the checkpoint's other preprocessing rules still apply; for example, 800 × 600 becomes 800 × 608. Near the limit, even a smaller image can change size because of the 32-pixel rounding: 977 × 1073 becomes 960 × 1056 under this budget, against 992 × 1088 under the checkpoint's own budget.

Small text can become blurry when a large image is reduced. Crop the relevant region, or raise the budget when fine detail matters. To restore the checkpoint's 16,777,216-pixel budget:

```bash
--mm-processor-kwargs '{"images_kwargs": {"size": {"longest_edge": 16777216, "shortest_edge": 65536}}}'
```

- An omitted or empty `--mm-processor-kwargs` selects the 1,048,576-pixel default. Explicit `size`, `max_pixels`, `min_pixels` or `do_resize` settings, including those inside `images_kwargs`, take precedence.
- The startup log reports the effective budget and its source: `Qwen4Exp image max_pixels=<value> source=<default|operator|request> do_resize=<value>`.
- The default changes only the image processor. Top-level size settings can also reach the video processor through HF's routing; use `images_kwargs` for an image-only change.
- Memory profiling uses the same budget: at most 1,024 image tokens per image by default, against 16,384 with the checkpoint's budget. A smaller image budget does not by itself guarantee a larger KV cache; encoder and KV memory also depend on video input and the measured GPU allocations.

## Video frames

The video loader samples **16 frames** per video by default. The reference launch states this explicitly with `--media-io-kwargs '{"video": {"num_frames": 16}}'`. An operator can set `num_frames` or `fps` there instead. When a trusted request overrides one of them, the other inherited field is cleared. `num_frames=-1` loads all frames only when that count fits the frame ceiling and the decode budget; this is checked before the frames are allocated.

The frame ceiling (`VLLM_MAX_VIDEO_FRAMES`, default 64) and the decode budget apply after these settings are merged. Video backends that cannot account for frames before decoding are rejected while the decode budget is enabled.

## Request budgets

These server environment variables bound the media in one request. Set a variable to zero to disable that ceiling. Requests cannot raise them.

| Variable | Default | Scope |
| --- | ---: | --- |
| `VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB` | 256 MiB | One remote response, including decompressed HTTP content and cached response bytes |
| `VLLM_MAX_MEDIA_REQUEST_BYTES` | 268435456 B (256 MiB) | All encoded bytes in the request, from HTTP, data URLs, files and download-cache hits |
| `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES` | 536870912 B (512 MiB) | All decoded image and sampled-video rasters in the request, counted at four bytes per pixel |
| `VLLM_MAX_MEDIA_REQUEST_ITEMS` | 32 | All fetched raw-media items in the request, across modalities |
| `VLLM_MAX_IMAGE_PIXELS` | 16777216 | Pixels per image or video frame, before resizing |
| `VLLM_MAX_VIDEO_FRAMES` | 64 | Sampled frames per video |

- The single-response limit stops one large download; the request limits stop many smaller inputs.
- Images are charged at their original size, before resizing. The default decode budget fits 16 frames of 3840 × 2160 (530,841,600 B) or several 12-megapixel images. With 16 images per request, each image can be at most 8,388,608 pixels (536,870,912 B / 16 / 4 B per pixel) before the request is rejected; raise `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES` if your clients send many large photos.
- The decode budget does not promise that a long source video fits the separate 256 MiB download limit.
- Repeated URLs and cache hits are charged again. Failed attempts keep their charges for the current request; a new request starts with a fresh budget.
- Size errors are validation failures: the server returns HTTP 400 and the client should not retry unchanged.

## Per-request overrides

Requests that carry a non-empty `mm_processor_kwargs` or `media_io_kwargs` are rejected with HTTP 400 by default. Enable `--trust-request-mm-kwargs` only when all API clients are trusted. Omitted values, `null` and empty dictionaries are accepted. The server-level `--mm-processor-kwargs` and `--media-io-kwargs` remain operator settings. The same policy applies to the render, tokenize, pooling and `run-batch` frontends; the local Python `LLM` API keeps accepting caller-controlled overrides.

## Image decoding

- EXIF orientation is applied when encoded images are loaded.
- Transparent images (RGBA, LA, palette transparency and PNG tRNS) are composited onto white before RGB conversion. This matches the Qwen example workflow's [`qwen_vl_utils.to_rgb`](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py); the generic Transformers RGB converter does not composite alpha. `rgba_background_color` in the image `--media-io-kwargs` selects another RGB color.

## Memory boundaries

The budgets above limit admitted and retained media data. They are not a hard ceiling on process memory.

- Remote bodies are streamed in 64 KiB chunks. `Content-Length` is checked before reading; missing lengths and gzip or deflate responses are counted while reading, and decompression output is bounded. Files and cached responses are also read incrementally, and a data URL's size is checked before base64 decoding.
- Images are charged before EXIF processing or raster loading. The common OpenCV and PyAV video loader checks the selected frame count and frame size before allocating frame arrays. JPEG frame sequences charge each image before loading and check the frame count before stacking.
- Video codecs can allocate internal reference frames and scratch buffers, and inaccurate container dimensions are detected only after decoding. Image conversion, frame stacking and joining encoded chunks also make temporary copies.
- Each concurrent request has its own budget. Image and video preprocessing runs in the API server process; see [System memory and storage](../docs/configuration.md#system-memory-and-storage) for a measured burst. Also bound HTTP request bodies, concurrent requests and process or container memory.
- Encoded audio bytes are counted, but audio decoding is outside these image and video limits.

## See also

- [Getting started](../docs/getting-started.md): the reference multimodal launch options
- [Multimodal launch options](../docs/configuration.md#multimodal-input)
- [Support and known limits](../docs/known-limits.md)
