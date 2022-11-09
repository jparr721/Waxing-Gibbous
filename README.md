# Waxing-Gibbous
🤫🤭 Neural Network Sag-Free Initialization???

## Running
Running this is super easy
```bash
$ ./initialize
$ ./run simulate
```
And you're done! This code also has some CLI integration, so just run

```bash
$ ./run --help
```
To see the available options.

### Training the Neural Network
The task of training the neural network has been taken care of via the built in options. To get a dataset to train, you just need to run the following command:
```bash
$ ./run generate <n_samples> --input-path <path> --output-path <path>
```
To see a full description of each of the options, run `./run generate --help`. Each sub-option has it's own sub-menu with a description.
Now that we have some data that the neural network knows how to ingest, we can begin to train our model. To train the model we just need to run
```bash
$ ./run train <input_file> <output_file>
```
And this will take the inputs and outputs and train the model on them. This will do two important additional things:
1. It will _save_ the neural network model so that way you can invoke it later
2. It will _show_ the loss plots so you can see how quickly your model converged.

The train script does _not_, however, give you any indication of the hyper-parameters that you used, make sure you write them down.