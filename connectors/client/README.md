# Image Segmentation Client

...for the network based [image segmentation server](../server/README.md). The client is written
in Python but could have been written in any (decent) programming language. Check the first
few lines of each file where (usually) a small help (or an example command) is left.

## Minimal Software Dependencies

Switch to local `pixi.toml` to obtain a minimalistic [Python environment](../../PIXI_ENVIRONMENTS.md)
that is large just enough for [this client](segment_image.py) to run well.

```bash
pixi shell --frozen --manifest-path connectors/client/pixi.toml
```

The environment basically consists of recent Python and the `requests` package, nothing more.

## The Segmentation Client

The [client Python script `segment_image.py`](segment_image.py) asks for an URL to a running server,
[name of a network model on that server](../server/README.md#server-naming-scheme), path to input
image to be segmented and a path to which the result should be stored. The files are expected to
be TIFF image formats.

The code reports times (delays) on its console, but the report cannot disambiguate how much
from a delay was spent in the images transfers and how much in the inference (segmentation) itself.

## The Image Round Trip Client

There's also a ["image back-and-forth" Python script `segment_image_RTT.py`](segment_image_RTT.py)
that needs URL to a running server and a path to input image to be sent around. The client code
uses the [server's round trip API](../server/README.md#image-round-trip-time), and (client) reports
obtained times (delays) on its console.

