## Resources and Setup Guide for GraphDTAx

### Resources Available:

- **README.md**: Main documentation file.
- **Data Files**:
  - `data/davis/folds/test_fold_setting1.txt`, `train_fold_setting1.txt`; `Y`, `ligands_can.txt`, `proteins.txt`
  - `data/kiba/folds/test_fold_setting1.txt`, `train_fold_setting1.txt`; `Y`, `ligands_can.txt`, `proteins.txt`
  - `data/bindingdb/folds/test_fold_setting1.txt`, `train_fold_setting1.txt`; `Y`, `ligands_can.txt`, `proteins.txt`
- **Source Codes**:
  - `create_data.py`: Script to create data in PyTorch format.
  - `utils.py`: Includes `TestbedDataset` for data creation and performance measures.
  - `training.py`: Script to train and test the GraphDTAx model.
  - Model Implementations:
    - `models/enhanced_gatgcn.py`
    - `models/enhanced_gat.py`
    - `models/SAGEConv.py`
    - `models/GPSConv.py`

### Step-by-Step Running Guide:

#### Step 1: Install Python Libraries

1. **Install Required Libraries:**
   - PyTorch Geometric, RDKit, and other dependencies. Use the following commands:

   ```bash
   conda create -n geometric python=3
   conda activate geometric
   conda install -y -c conda-forge rdkit
   conda install pytorch torchvision cudatoolkit -c pytorch
   pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
   pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
   pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
   pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
   pip install torch-geometric
   ```

#### Step 2: Create Data in PyTorch Format

2. **Generate Data Files**:
   - Activate the environment and run the data creation script:

   ```bash
   conda activate geometric
   python create_data.py
   ```

   - This will generate CSV files (`kiba_train.csv`, `kiba_test.csv`, `davis_train.csv`, `davis_test.csv`, `bindingdb_test.csv`, `bindingdb_train.csv`) and save them in the `data/` directory. It also prepares `.pt` files for PyTorch.

#### Step 3: Train and Test the Prediction Model

3. **Model Training and Testing**:
   - Use the following command to train models and test their performance:

   ```bash
   conda activate geometric
   python training.py 0 0 0
   ```

   - Parameters:
     - CUDA device index (`0` for 'cuda:0', `1` for 'cuda:1'). Adjust the `cuda_name` in the script based on your system configuration.

   - The script returns model files (e.g., `model_GraphSAGENet_davis.model`) and result files (e.g., `result_GraphSAGENet_davis.csv`) representing the best MSE achieved during the training.

### Conclusion

Follow these steps to set up and run the GraphDTAx model training and testing using the provided scripts and data. Ensure all dependencies are installed correctly to avoid any issues during the execution of the scripts."# GraphDTAx" 
