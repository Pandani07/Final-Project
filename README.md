# Final-Project

Steps to run:

1. Clone repo
2. Install requirements
3. Navigate to project folder in cmd
4. run "python app.py"

Use Spyder IDE through Conda navigator to make changes/ run the model.py file

When you run python app.py in the command prompt, youll get these warnings. They are warnings, not errors so ignore

* *W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2021-05-20 22:24:47.868714: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: LAPTOP-0U1MGGAN
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: LAPTOP-0U1MGGAN
I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Tensorflow/ LSTM model loaded <tensorflow.python.keras.engine.sequential.Sequential object at 0x000002CDBE0C4E50>* *

Server will be running on http://127.0.0.1:5000/ 


## We have added pikl files to gitignore due to storage constraints. Goodle drive link with all pikl files has been provided. Extract into a folder called 'pikl_files' in project directory. Pikl model for NIFTY 50 is the 'pickle_model.pkl' in project root directory. Delete pikl_files folder whenever you are pushing into Github.

## Project Directory structure:

final-project
<ul>
   <li>pikl_files</li>
   <li>static</li>
   <li>templates</li>
   <li>app.py</li>
   <li>indicators.p</li>
   <li>mldlfile.py</li>
   <li>model_100.h5</li>
   <li>neuralnetwork.py</li>
   <li>pickle_model.pkl</li>
   <li>NIFTY 50.csv</li>
   <li>preprocessing.py</li>

</ul>

