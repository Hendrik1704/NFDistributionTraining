## Example Usage

This README file provides instructions on how to use the code in the `tests` 
directory for the tests that are not described in the main README file already.

### Rosenbrock Banana Continue Training File
This example shows how to continue training a NF model from a saved model file.
First, generate the training and validation data by running the following script:
```bash
python3 generate_training_validation_data_Rosenbrock_Banana.py
```
This will create the training and validation data files. Then, run the 
first training script, that will be interrupted after a set time:
```bash
python3 Rosenbrock_Banana_NF.py
```
This will start the training of the NF model and after a set time (45 seconds by default)
it will simulate a Ctrl+C interrupt to stop the training. The NF model state will be saved
to a file called `normalizing_flow_model_Rosenbrock_Banana.pkl` in the 
`tests/Rosenbrock_Banana_Continue_Training_File` directory.

Finally, run the second training script to continue the training from the saved model file:
```bash
python3 Rosenbrock_Banana_NF_Continue_From_File.py
```
This will load the saved NF model and continue the training for another set number of steps.