# Image Segmentation on Command Line

...using the cellpose architecture. This code is written in Python because it works
directly with the cellpose API, which is in Python too.

## cellpose-centric Software Dependencies

Switch to local `pixi.toml` to obtain a minimalistic [Python environment](../../PIXI_ENVIRONMENTS.md#cellpose-env)
that is large just enough to host the cellpose original library code plus I/O API etc.

👉 Note that two [versions of cellpose](../../PIXI_ENVIRONMENTS.md#cellpose-env) are supported in
both the [segmentation](segment_image.py#L4-L15) and [training](train_two_folders.py#L4-L15) scripts.

```bash
# default is cellpose v4 with GPU
pixi shell --frozen --manifest-path connectors/cellpose/pixi.toml

# another example can be cellpose v3 with CPU:
cd connectors/cellpose
pixi shell --frozen -e cpu-v3
```

## The Segmentation Clients

In this folder is a [workhorse Python client `segment_image.py`](segment_image.py) file,
usage of which is, however, also exemplified in [Windows batch file](segment_image.bat) as
well as in [Linux/Mac shell script](segment_image.sh).

The latter two OSes-native scripts are collecting parameters, starting Python environment
(with the local pixi, see above) in which the `segment_image.py` is executed. The workhorse
does the job: loads the model, loads the input image, [normalizes it](segment_image.py#L30),
applies the model on it, and saves the result.

## The cellpose Training Code

Here, only the Python [heavy-lifting script is provided `train_two_folders.py`](train_two_folders.py).
Please, consult the `.bat` and `.sh` OS-native scripts and how they start the Python
environment and then the segmentation script itself to see how similar OS-native scripts
can be easily created.

The Python training script requires a name of the output model file that is ideally [in the server understandable
format](../server/README.md#models-naming-scheme), batch size and number of epochs, then
path to a folder with the training image pairs, and another path to a folder with image
pairs for the model evaluation during the training.

Image pairs are established from every discovered `raw*.tif` file by finding a matching
file `mask*.tif` to each. Both files *must be* thus in the same folder. The pairing
happens through file names, for a `rawQQQQ.tif` the code searches for `maskQQQQ.tif` where
the `QQQQ` can be arbitrary string, and of course the same for both files to make the
connection that the two files belong together. The `raw...` file must show the raw image,
the one obtained on a microscope. While the `mask...` file must show its corresponding
instance segmentation. Both are expected to be 2D images.

