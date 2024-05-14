## D-AMP via Diffusion

This project implements a minimal runnable example of the paper ``D-AMP via Diffusion''. You may need to download the ImageNet and Berkeley Segmentation Dataset for a more complete evaluation.  

> For AMP algorithm benchmarks, please refer to [AMP-Toolbox](https://github.com/ricedsp/D-AMP_Toolbox). 

<br />

### Inference 
- 1. `amp_inference` implements the inference process. 
    - Install Python packages
    ```bash
    pip3 install -r requirements.txt
    ```

- 2. A pretrained model is available here [(model.pt)](https://drive.google.com/file/d/1BdxBZPDfeBxbLNA8kpJXeqkacYMEyoOw/view?usp=drive_link). Download it and put it under `model_zoo`. 
- 3. Use `python3 main.py` to perform inference. The progressive results will be stored under `experiments` folder. 
- 4. Modify `config/config.yaml` to change hyperparameters and setups. For example, one can change the number of iterations in AMP to $10$ by modifying the entry: 
`amp_iters: 10`. 

<br />

### Train a Neural Network Denoiser

- 1. `amp_train` implements the training process.
  - Install Python packages
    ```bash
    pip3 install -r requirements.txt
    ```
- 2. `amp_train/data` folder contains the preprocessed dataset. 
  - 2.1 `data/val/gt` contains the ground truth images. Each subfolder (`0`) means the quantized $\sigma$, which can be equivalently treated as the label. This label is required as the neural network input. 
  - 2.2 `data/val/noise` contains the noisy images. Again, each  subfolder (`0`) means the quantized $\sigma$. Each folder pairs with a counterpart under `val/gt`. 
  - 2.3 `data/gt/datapair.dat` and `data/noise/datapair.dat` define the path of each training image. This data structure is crafted to help SGD mini-batch sampling with loading all samples in the memory. 
- 3. Modify `config/config.yaml` to change hyperparameters and setups. For example, one can change the learning rate to 0.01 by modifying the entry to `lr: 1.e-2`. 

