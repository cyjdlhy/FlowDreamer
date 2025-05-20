# FlowDreamer

This is the official implementation of **FlowDreamer: Exploring High Fidelity Text-to-3D Generation Via Rectified Flow**.

### [Project Page](https://vlislab22.github.io/FlowDreamer/) | [Arxiv Paper](https://arxiv.org/abs/2408.05008v3)


![FlowDreamer Cover](https://github.com/cyjdlhy/assets/blob/main/FlowDreamer/cover.png)  
![FlowDreamer Video Demo](https://youtu.be/NCw2Qi0zoIk?si=xJamrWwk3yaULKFj)
---

### Installation

To get started with FlowDreamer, follow the installation instructions below:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/cyjdlhy/FlowDreamer.git
    cd FlowDreamer
    ```

2. **Create a Conda Environment**

    ```bash
    conda create -n FlowDreamer python=3.9.16 cudatoolkit=11.8
    conda activate FlowDreamer
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    pip install submodules/diff-gaussian-rasterization/
    pip install submodules/simple-knn/
    ```

4. **Download the Model**

    - Modify the `model_key` path in the `configs/base.yaml` file to point to Stable Diffusion 3 model (e.g., [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)).
    
5. **Run**

    ```bash
    bash train.sh
    ```

---

### Acknowledgements

This project is built upon the work of several excellent research projects and open-source contributions. A big thank you to all the authors for sharing their work!

- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Diff-Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [LucidDreamer](https://github.com/EnVision-Research/LucidDreamer.git)
- [Point-E](https://github.com/openai/point-e)

---

### Citation

If you find this project useful in your research, please consider citing our paper:

