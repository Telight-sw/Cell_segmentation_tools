# Image Segmentation Server

In this folder, one can find a very basic Python server based on the project `fastapi`.
The framework uses annotations/decorations around functions using which HTML addresses
are defined to which the server responses, and using which parameters can be provided etc.

The framework also automatically supports common specific URLs like  
👉 `/docs` that lists the currently provided URLs. This is very helpful to know, btw.

The server supports several segmentation methods to be applied on the images.
A segmentation method is understood to be a particular segmentation network (architecture)
*and* its [model file](../../SLURM/testing_evaluation/common_tools.py#L42).
The model file thus needs to be named properly because the matching architecture,
that the server will use, is [read from the model file name](segmentation_methods.py#L30-L35).

The server comes preloaded with *no model*, not even the networks' default ones
are available ("reachable" to be more precise).

<a name="server-layout"></a>
## Models, Models Folder, and the Server

The server is instructed to look for models in a folder. The folder with models files
is [defined in the source code](server.py#L15), must be defined prior the server start,
and cannot be changed at runtime.

<a name="models-naming-scheme"></a>
The models files must be named according to this pattern:  
`trainingDatasetNickname.trainingConfiguration.networkName.model`  
where

- `trainingDatasetNickname` is any (human-readable) string to indicate for which kind of data the model has been created,
- `trainingConfiguration` is an auxiliary string that encodes e.g. training dataset size, batch size, epochs,
- `networkName` is the name of the underlying network architecture,
- `model` is mandatory suffix.

<a name="server-naming-scheme"></a>
The server will reference each model file under its name `networkName.trainingDatasetNickname`.
The idea is that if, for example, a model `nucleiQWE.t600_bs8_e500.instanseg.model` has been created,
it is in the server referred to as `instanseg.nucleiQWE`. Such name is used in the server's URLs!

👉 URL`/segmentation_2D/list_available_methods` lists models visible to the server, and  
👉 URL`/segmentation_2D/update_available_methods_list` makes the server to re-read the content of its models folder.

So, yes, the models can be changed, removed, or added to the running server, and the server can
be notified... For performance reasons, the server doesn't actively monitor the models folder.

<a name="encrypted-models"></a>
### Encrypted Models

There exists [an modification of the server that assumes the model files are encrypted](server_with_encryption.py#L39).

In this case,

- a symmetric [cryptography key should be obtained](segmentation_encrypted_methods.py#L70-L72),
- an original [model file should be encrypted](segmentation_encrypted_methods.py#L83),
- and the encrypted file should be placed into the server models folder,
- and `server_with_encryption.py` should be started with `fastapi` instead,
- and its URL-based API is slightly different (as encryption=decryption key needs to be provided).


## Starting the Server

A server-enabled `pixi.toml` is available in this folder too. One, thus, has to start
the Python environment from this folder:

```bash
cd connectors/server
pixi shell

# -- or --
pixi shell --manifest-path connectors/server/pixi.toml
```

and then

```
fastapi run server.py

# -- or --
fastapi run server_with_encryption.py
```

Use `fastapi dev server` (instead of `run` command) if you are developing/changing
the server source file(s).

If all goes well, the console will report (among other stuff) something like this:

```
   server   Server started at http://127.0.0.1:8000
   server   Documentation at http://127.0.0.1:8000/docs
```

It is possible to start the server listening on (bound to) a particular IP address and port:

```bash
uvicorn --host 0.0.0.0 --port 8765 server:app

# -- or --
uvicorn --host 0.0.0.0 --port 8765 server_with_encryption:app
```

This command basically
[starts the production server backend directly](https://fastapi.tiangolo.com/deployment/manually/#asgi-servers).  
Note, that [somewhat similarily a https server could be started](https://fastapi.tiangolo.com/deployment/https/).


Once again, opening the *Documentation URL* (URL`/docs`) in a web browser gives
an overview of what the server currently supports and how it is used.


## Submitting 2D Image for Segmentation

For this, the server supports URLs of two kinds:

- Post serialized TIFF image, literally exactly as it can be found on a hard drive.
  The server will respond back with another serialized TIFF image.
- Post a sequence of 32-bit float values, the server will reply with a sequence of 16-bit unsigned ints.

The following screenshot may explain it all :)

![Screenshot showing an example of the /docs URL with manual to some existing functionality](../../doc/imgs/server_API.png)

Notice the documentation gives example `curl` commands that will not work straight away
as they are referring to test data, or test TIFF files, that of course one doesn't have
on her hard drive. The example shell commands would need to be slightly adapted.

[Python client example is available as well.](../client)


## Stopping the Server

For the sake of being complete, to stop the server one has to come back
to the console where the server has been started, and press `Ctrl+C`.

<a name="image-round-trip-time"></a>
## Understanding Performance

It is useful to start the server on a console/terminal display of which can be
(re-)accessed anytime. That is because the server reports various information to its
console, and also a time spent per various actions is reported too. This can be used
to understand possible bottlenecks when server is showing delays.

Furthermore, the server's HTML API offers a command in which the image data is only
arrived to the server and sent back -- a round trip of the pixel data, no network
inference is carried on. This can be used to see what delays the network can cause.

