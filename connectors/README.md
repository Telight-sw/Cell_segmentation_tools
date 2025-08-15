# Making the Networks Available in Another Software

There are basically two general options. Either the network code is [linked to the host
program](#linked-library) (and becomes part of the program at runtime), or the host
program accesses the networks code through some adapter code or translational layer.
Example of the latter can be a [network stack](#client-server) or some (ideally
POSIX-compatible) interprocess communication like [Appose](https://github.com/apposed)
that both marshal the commands and data between the host and networks programs.

This is exactly the purpose of this folder: Introduce a blend of example and
close-to-production codes that offer means to access the networks from this project.

- A command-line solution in [`cellpose`](cellpose)
- A [`client`](client) to a network-attached [`server`](server)

[See below for more details.](#accessor-implementations)

<a name="linked-library"></a>
## Linked Library

The network code needs to be written in a language compatible with that of the host
program. For Telight (and SophiQ) this would mean to re-implement the networks in C++
as no network (at least, none from the tested ones) today is developed in anything
different than Python.

#### Pros

Having the network implementation in C++ is very efficient solution *at runtime* as both
the host program as well as the network code see the same memory and can share the image data
directly. Next to the "data link", also the "command link" (which here is calling the
linked-library API) comes with little-to-none communication overhead.

The network's software stack can be initiated, which is relative a time consuming
operation, just once at the host program start-up. During subsequent usages of the network,
only the inference code is (swiftly) executed.

#### Cons

Since the network code here is re-implemented from its native mainstream codebase,
upgrading to a newer or another architecture could be (repetitively) non-trivial.
Note that some [networks today utilize also non-neuronal-network code](../segmentation/README.md#wrapped-networks),
that said, a code that is not based on the Torch `nn.Module` base class.

<a name="library-in-another-compiler"></a>
#### Remarks

Directions to alleviate the above seem available by using automated deployment systems
like [ONNX](https://onnx.ai/), which represents the network in some "abstract" code that
can be instantiated to a user-desired framework and compiler (but libtorch is not listed
among the supported combinations). Another option could the
[TorchScript](https://docs.pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html),
which promises C++ (and thus libtorch), also [more own conclusions from reading the
docs](../py-to-cpp/README.md). For example, Instanseg network is utilizing this, but
AFAIK it is the only one from those considered here. Note also that ONNX under its hood
seems to be utilizing the TorchScript. Finally, I saw Internet discussions that claim
that PyTorch objects exported natively with PyTorch API should theoretically be loadable
in libtorch using its API, but in this project this path hasn't been explored at all.
Nevertheless, this could easily be only half of story as the modern networks typically
feature some [inevitable pre- and post-processing (CPU) stages](../segmentation/README.md#wrapped-networks)
that would have to be rewritten to C++ (but potentially only once for a given network
package).

Another remark is that this solution is by its design not directly available for a remote
computation, e.g., when more powerful network-connected hardware is available. However,
with reasonable amount of effort, the remote communication can added, e.g., by embedding
a "forwarding" layer.


<a name="client-server"></a>
## Client-server Approach

The SophiQ (or any other program) would act as a client to a network-attached server that
offers segmentation services to the clients. The clients would send the image and an
information which model (and transitively which network architecture) to use over the
network, and the server would process (segment) the image and return the result.
The server can run either on the same machine, or some remote one.

#### Pros

Since the server implementation is unconstrained, especially not constrained to what
SophiQ requires and can be linked to, the server can be implemented in the programming
language of the network(s) it is serving. It can be merely a wrapping code around API
calls to the libraries of the original networks, and there are good frameworks for exactly
that.

It is expected that, with every upgrade (or new addition) of a network, the server code is
easily updated (or new wrapping code is added) to the API of that network. The network
especially remains intact, as it comes tuned from its authors, and in its native computing
environment.

The networks' original libraries and their full software stack can be initiated, which is
relative a time consuming operation, just once at server start-up. During subsequent
usages of the networks, only the inference code is (swiftly) executed.

#### Cons

Because of the communication overhead and also because the data and results need to
transferred, this solution is clearly less-efficient at runtime compared to the [linked
library](#linked-library) approach.

The server code needs to be maintained, which is an extra, development effort. In
production, SophiQ will need to manage starting/stopping/monitoring the server instance
(which should be possible with POSIX API of the operating systems, [Linuxes have open
subsystems dedicated for exactly this kind of services such as
`systemd`](https://en.wikipedia.org/wiki/Systemd)).

<a name="client-server-remarks"></a>
#### Remarks

This paradigm opens up also a business opportunity for the company. Owing to the
network-based design, the customers can have their images processed remotely, outside
their computer. That way they can reach, e.g., "early-access" models for testing (and the
company could sample better what images are the users testing on), or simply access more
powerful hardware (by temporarily renting company's hardware) for inference or especially
for training/adopting models. Note that training the networks generally requires more
resources than the inference, and it could become an interesting model for some customers
to buy lower-end hardware for everyday use (inference) and rent another hardware for the
training. This is assuming a not so difficult case that the server would be able to
collect data for training/adopting models.

Consider also that not necessarily must the clients rent Telight-provided powerful
servers, imagine they have a good workstation available within their (research) group that
could be used for inference (rather than laptops with SophiQ). This is a great added
value to the users that costs Telight nearly no development time.

<a name="accessor-implementations"></a>
## Implementation

Two variants to the [client-server](#client-server) approach have been implemented.

### Command-line cellpose Scripts

[The first solution](cellpose/README.md) is a simple command-line script
that is assumed to be POSIX-called (executed as an independent process) from SophiQ. The
image data would need to be written to some local path (to a drive), and later read from
another path after the segmentation is over. The network cellpose is always used but its
model can be changed by providing the path to a stored file with the model. Altogether,
a couple of path parameters must be provided to the script.

In the same folder is available also a [like-minded script that allows to
train](cellpose/train_two_folders.py) cellpose over provided folders with
training and validation data, and using given [training parameters](../doc/README.md#training-parameters).
The script will eventually create a model file under the provided path (parameter to the script).

### Network-based Generic Server and an Example Client

[The network-based client-server solution](server/README.md) has been
implemented too. It exists also in a version that [operates on encrypted
models](server/README.md#encrypted-models). Briefly, the server is
pointed to a folder with models that are named specifically such that a network
architecture can be understood from the model name, and the server can segment the given
images using several network architectures. The server can be also notified during
runtime of added (or removed) model files. And the server implements a simple system
of HTML URLs using which the client requests the segmentation service.

A [demo client in Python](client/README.md) is also available. Besides
demonstrating the viability of the client-server solution, especially when [time
demand](server/README.md#image-round-trip-time) is considered, the client also
shows that only a [limited number of supporting libraries](client/pixi.toml#L16-L17)
is required.

