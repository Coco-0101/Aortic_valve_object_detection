# AI CUP aortic value object detection
## Conda Envrionment
### Create a New Environment
```bash
conda create -p /home/vcpuser/netdrive/Workspace/conda_envs/AI_CUP python=3.10
```

>`-p` specifies the full installation path (prefix).  
Recommended when using containers or mounted drives to keep environments persistent and organized.

---

### Create Environment from Existing YAML
```bash
conda env create -f AI_CUP.yaml -p /home/vcpuser/netdrive/Workspace/conda_envs/AI_CUP
```

Activate the environment:
```bash
conda activate /home/vcpuser/netdrive/Workspace/conda_envs/AI_CUP
```

---

###  Set Environment Prompt
If you want the terminal prompt to show only the environment name:
```bash
conda config --set env_prompt '({name})'
```

Then reload the environment:
```bash
conda deactivate
conda activate /home/vcpuser/netdrive/Workspace/conda_envs/AI_CUP
```

After activation, the prompt will display `(AI_CUP)` instead of the full path.

---

###  Export Environment
#### Option 1: Export by name (`-n`)
```bash
conda env export -n AI_CUP > AI_CUP.yaml
```

#### Option 2: Export by path (`-p`)
```bash
conda env export -p /home/vcpuser/netdrive/Workspace/conda_envs/AI_CUP > AI_CUP.yaml
```

> Use the `-p` option if the environment was created using a path and does not have a registered name.

---

### 💡 Tips
To make your exported `AI_CUP.yaml` more portable across systems, remove the following line from the YAML file:
```yaml
prefix: /home/vcpuser/netdrive/Workspace/conda_envs/AI_CUP
```
