# TinyDiffusion
Optimized Diffusion for Edge Devices

Add the project root ie, Folder containing this README to PYTHONPATH whichever way you want. One way would be to create a .env and write the following in it
```
PYTHONPATH=\full\path\to\projectroot
```
And place this .env file in the project root. Works for VS Code.

Another option would be to run `$env:PYTHONPATH = \full\path\to\projectroot` in powershell to set the env variable and then run the scripts.

Install pytorch, torchvision via `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118` - Conda doesnt install GPU version on Windows.

##### Sanity

Before committing changes run `pre-commit run --all-files` or `pre-commit run --file <file1>, <file2> ...`
