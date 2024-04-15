# OCR Agent from scratch:
Let's build a custom OCR agent with specific requirements:
a.) high privacy (sensitive data processing);
b.) ability to find and read random alphanumeric codes (identification) with high precision;
c.) ability to learn from interactive feedback.

We won't be using any OCR libraries or services.

We extract layout features using `OpenCV` and train our local nearsighted (char-level) agent to identify each patch and traverse the patches properly to aggregate the text and tables data in a meaningful way (we build our ti.
Modern laptop should be able to handle this project.

As the example workload we used tax-forms: the problem stated as visual layout understanding enabling high-precision data extraction from the multi-page scanned documents (set of page-images) with high number of different classes to recognize, and sensitive user data which must be properly protected.

![Reader-walk record with matplotlib](./notebooks/assets/reader-walk.gif)


### Content

* [Data](./notebooks/Data-Sources.ipynb)
* [Utilities](./notebooks/Data-Processing.ipynb)
* Extract layout features and visual tokens
    * [Cells and grid-lines (tables)](./notebooks/Data-Extraction-1.ipynb)
    * [Text-lines, word-level objects, char-level tokens](./notebooks/Data-Extraction-2.ipynb)
* Generate training data
    * [Labeling](./notebooks/Data-Extraction-3.ipynb)
    * [Pipeline](./notebooks/Data-Pipeline.ipynb)
    * [Datasets](./notebooks/Datasets.ipynb)
* Model architecture
    * [Visual encoder, generative and discriminative heads](./notebooks/Model-Backbone.ipynb)
    * [Unsupervised and semi-supervise pretraining](./notebooks/Model-Pretraining.ipynb)
    * [Supervised multi-task training](./notebooks/Model-Training.ipynb)
* Traversal strategies
    * [Layout traversal](./notebooks/Traversal-Layout.ipynb)
    * [Text aggregation](./notebooks/Traversal-Text.ipynb)
    * [Form extraction and validation](./notebooks/Traversal-Form.ipynb)
* Reader Agent
    * [Wire language model in](./notebooks/Agent-LM.ipynb)
    * [Set RAG utilities](./notebooks/Agent-RAG.ipynb)
    * [Define FSM](./notebooks/Agent-FSM.ipynb)
    * [Reinforcement learning setup](./notebooks/Agent-RL.ipynb)
* [Leverage synthetic training data](./notebooks/Data-Gen.ipynb)
* Optimization for production


### Environment

    root/
    ├── Dockerfile
    ├── requirements.txt
    ├── init.cnf               -- example of env-configuration file
    ├── ...
    │
    ├── notebooks/             -- jupyter notebooks server root
    │   ├── ...
    │   ├── data/
    │   │   ├── ...
    │   │   ├── forms/         -- original PDF files (multi-page)
    │   │   ├── images/        -- images of pages
    │   │   ├── content/       -- extracted textual content and layout data
    │   │   ├── inputs/        -- extracted form inputs data
    │   │   ├── training/      -- extracted labeled samples
    │   │   ├── feedback/      -- human labeled samples
    │   │   └── ...
    │   ├── ...
    │   ├── models/            -- trained models
    │   ├── output/            -- training outcome: history and plots
    │   ├── scripts/           -- local python library
    │   ├── runs/              -- tensorboard logs
    │   └── ...
    │   
    └── app/                   -- frontend for human interaction
        └── ...    



```python
!python --version
```

    Python 3.10.13



```python
!nvidia-smi
```

    Wed Mar 27 21:46:14 2024       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.147.05   Driver Version: 525.147.05   CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
    | N/A   49C    P0    28W / 108W |    272MiB /  8192MiB |      6%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A       809      G   /usr/lib/xorg/Xorg                 54MiB |
    |    0   N/A  N/A      1321      G   /usr/lib/xorg/Xorg                107MiB |
    |    0   N/A  N/A      1449      G   /usr/bin/gnome-shell               41MiB |
    |    0   N/A  N/A     14167      G   ...on=20240325-180218.632000       51MiB |
    +-----------------------------------------------------------------------------+



```python
import cv2; print(cv2. __version__)
```

    4.9.0



```python
import torch; print('GPU' if torch.cuda.is_available() else 'CPU'); print(torch.__version__)
```

    GPU
    2.2.2+cu121


